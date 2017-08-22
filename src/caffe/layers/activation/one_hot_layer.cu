
#include <vector>

#include "caffe/layers/activation/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void kernel(int count, int channels,int spatial_dim, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		int label_index = in[i];
		out[(n*channels+label_index)*spatial_dim+s] = 1;
	}
}

template <typename Dtype>
void OneHotLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	caffe_gpu_set(top[0]->count(),Dtype(0),top[0]->mutable_gpu_data());
	kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),classes_,height*width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void OneHotLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	
}

template <typename Dtype>
void OneHotLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(OneHotLayer);
}  // namespace caffe
