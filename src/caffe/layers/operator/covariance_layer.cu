
#include <vector>

#include "caffe/layers/operator/covariance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void kernel(int count, int channels,int height,int width, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		out[i] = in[i];
	}
}

template <typename Dtype>
void CovarianceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels, channels, height*width,
														(Dtype)1., bottom[0]->gpu_data() + bottom[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(Dtype)0., top[0]->mutable_gpu_data() + top[0]->offset(n));		
	}
	caffe_gpu_scal(top[0]->count(),Dtype(1)/Dtype(height*width),top[0]->mutable_gpu_data());	
	

}

template <typename Dtype>
void CovarianceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  

  caffe_gpu_scal(top[0]->count(),Dtype(1)/Dtype(height*width),top[0]->mutable_gpu_diff());		
	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels, height*width, channels,
														(Dtype)1., top[0]->gpu_diff() + top[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(Dtype)0., bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));		
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels, height*width, channels,
														(Dtype)1., top[0]->gpu_diff() + top[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(Dtype)1., bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));	
	}

}
template <typename Dtype>
void CovarianceLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(CovarianceLayer);
}  // namespace caffe
