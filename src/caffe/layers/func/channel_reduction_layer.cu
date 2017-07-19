
#include <vector>

#include "caffe/layers/func/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void forward_kernel(int count, int channels,int spatial_dim, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;

		out[i] = max(in[(n*channels*2+c)*spatial_dim+s],in[(n*channels*2+channels+c)*spatial_dim+s]);
	}
}
template <typename Dtype>
static __global__ void backward_kernel(int count, int channels,int spatial_dim, const Dtype * diff_out, const Dtype *in, Dtype *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		if (in[(n*channels*2+c)*spatial_dim+s] > in[(n*channels*2+channels+c)*spatial_dim+s])
		{
			diff_in[(n*channels*2+c)*spatial_dim+s] = diff_out[i];
			diff_in[(n*channels*2+channels+c)*spatial_dim+s] = Dtype(0);
		}
		else
		{
			diff_in[(n*channels*2+c)*spatial_dim+s] = Dtype(0);
			diff_in[(n*channels*2+channels+c)*spatial_dim+s] = diff_out[i];
		}
	}
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = top[0]->num();
  int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
	
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = top[0]->num();
  int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
  
	backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelReductionLayer);
}  // namespace caffe
