
#include <vector>

#include "caffe/layers/func/de_concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void deconcat_forward(int count,int channels, int i_channels, int cur_channels,int spatial_dim,
																const Dtype * b, Dtype *a)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / i_channels;
		int c = i / spatial_dim % i_channels;
		int s = i % spatial_dim;
		
		a[i] = b[(n*channels+cur_channels+c)*spatial_dim+s];		
	}
}

template <typename Dtype>
static __global__ void deconcat_backward(int count,int channels, int i_channels, int cur_channels,int spatial_dim,
																const Dtype *a, Dtype *b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / i_channels;
		int c = i / spatial_dim % i_channels;
		int s = i % spatial_dim;
		
		b[(n*channels+cur_channels+c)*spatial_dim+s] = a[i];
	}
}

template <typename Dtype>
void DeConcatLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	int cur_channels = 0;
	for (int i =0; i < top.size();i++)
	{
		int i_channels = top[i]->channels();
		deconcat_forward<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[i]->count(),channels,i_channels,cur_channels,height*width,
					bottom[0]->gpu_data(), top[i]->mutable_gpu_data());		
		cur_channels += i_channels;
	}
}

template <typename Dtype>
void DeConcatLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	int cur_channels = 0;
	for (int i =0; i < top.size();i++)
	{
		int i_channels = top[i]->channels();
		deconcat_backward<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[i]->count(),channels,i_channels,cur_channels,height*width,
					top[i]->gpu_diff(), bottom[0]->mutable_gpu_diff());		
		cur_channels += i_channels;
	}
}
template <typename Dtype>
void DeConcatLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(DeConcatLayer);
}  // namespace caffe
