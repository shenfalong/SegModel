
#include <vector>

#include "caffe/layers/activation/repmat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void repmat_kernel(int count, int channels,int height,int width, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		out[i] = in[n*channels+c];
	}
}

template <typename Dtype>
void RepmatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[1]->height();
  int width = bottom[1]->width();
	
	repmat_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height,width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void RepmatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[1]->height();
  int width = bottom[1]->width();
  
  caffe_gpu_set(one_multiplier_->count(),Dtype(1),one_multiplier_->mutable_gpu_data());
  
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num*channels, 1, height*width,
														(Dtype)1., top[0]->gpu_diff() , one_multiplier_->gpu_data(),
														(Dtype)0., bottom[0]->mutable_gpu_diff());  
}
template <typename Dtype>
void RepmatLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}
INSTANTIATE_LAYER_GPU_FUNCS(RepmatLayer);
}  // namespace caffe
