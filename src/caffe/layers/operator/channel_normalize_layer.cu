#include <vector>
#include "caffe/layers/operator/channel_normalize_layer.hpp"
#define BN_EPS Dtype(1e-5)

namespace caffe {
//---------------------------------------------------
#if 0
template <typename Dtype>
static __global__ void compute_norm(int count, int channels, const Dtype *in, Dtype *norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		Dtype sum = 0;
		for (int c=0; c<channels; c++)
			sum += in[(n*channels+c)*spatial_dim+s]*in[(n*channels+c)*spatial_dim+s];
		sum /= Dtype(channels);	
		norm[i] = sqrt(sum+BN_EPS);
	}
}
template <typename Dtype>
static __global__ void forward_kernel(int count, int channels, const Dtype *in, const Dtype *norm, Dtype * out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		out[i] = Dtype(0.02) * in[i] / norm[n*spatial_dim + s];
	}
}
//---------------------------------------------------
template <typename Dtype>
static __global__ void compute_diff_norm(int count, int channels, const Dtype *diff_out, const Dtype *in, const Dtype *norm, Dtype *diff_norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		Dtype sum = 0;
		for (int c=0; c<channels; c++)
			sum += diff_out[(n*channels+c)*spatial_dim+s]*(-in[(n*channels+c)*spatial_dim+s])/(norm[i]*norm[i]);
		diff_norm[i] = Dtype(0.02) * sum;
	}
}
template <typename Dtype>
static __global__ void backward_kernel(int count, int channels, const Dtype *diff_out, const Dtype * in, 
																const Dtype *norm, const Dtype *diff_norm, Dtype * diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		diff_in[i] = Dtype(0.02) * diff_out[i]/norm[n*spatial_dim+s] + diff_norm[n*spatial_dim+s]*in[i]/norm[n*spatial_dim+s]/Dtype(channels);
	}
}
#endif
//--------------------------------------------------
template <typename Dtype>
void ChannelNormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
#if 0
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	compute_norm<Dtype><<<num, CAFFE_CUDA_NUM_THREADS>>>
	(channels,bottom[0]->gpu_data(),norm_.mutable_gpu_data());
	
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,bottom[0]->gpu_data(),norm_.gpu_data(),top[0]->mutable_gpu_data());
#endif	
}

template <typename Dtype>
void ChannelNormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
#if 0
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<Dtype><<<num, CAFFE_CUDA_NUM_THREADS>>>
	(channels, top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_diff());
  
  backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_diff(),bottom[0]->mutable_gpu_diff());
#endif	
}
template <typename Dtype>
void ChannelNormalizeLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
#if 0
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<Dtype><<<num, CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels, bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_sec_diff());
  
  backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels, bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
#endif	
}
INSTANTIATE_LAYER_GPU_FUNCS(ChannelNormalizeLayer);
}  // namespace caffe
