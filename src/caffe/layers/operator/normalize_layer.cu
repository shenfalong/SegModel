#include <vector>
#include "caffe/layers/operator/normalize_layer.hpp"
#define BN_EPS Dtype(1e-5)

namespace caffe {
//---------------------------------------------------
template <typename Dtype>
static __global__ void compute_norm(int count, int channels,int spatial_dim, const Dtype *in, Dtype *norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		Dtype sum = 0;
		for (int c=0; c<channels; c++)
			sum += in[(n*channels+c)*spatial_dim+s]*in[(n*channels+c)*spatial_dim+s];
		norm[i] = sqrt(sum+BN_EPS);
	}
}
template <typename Dtype>
static __global__ void forward_kernel(int count, int channels,int spatial_dim, const Dtype *in, const Dtype *norm, Dtype * out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		out[i] = in[i] / norm[n*spatial_dim + s];
	}
}
//---------------------------------------------------
template <typename Dtype>
static __global__ void compute_diff_norm(int count, int channels,int spatial_dim, const Dtype *diff_out, const Dtype *in, const Dtype *norm, Dtype *diff_norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		Dtype sum = 0;
		for (int c=0; c<channels; c++)
			sum += diff_out[(n*channels+c)*spatial_dim+s]*(-in[(n*channels+c)*spatial_dim+s])/(norm[i]*norm[i]);
		diff_norm[i] = sum;
	}
}
template <typename Dtype>
static __global__ void backward_kernel(int count, int channels,int spatial_dim, const Dtype *diff_out, const Dtype * in, 
																const Dtype *norm, const Dtype *diff_norm, Dtype * diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		diff_in[i] = diff_out[i]/norm[n*spatial_dim+s] + diff_norm[n*spatial_dim+s]*in[i]/norm[n*spatial_dim+s];
	}
}
//--------------------------------------------------
template <typename Dtype>
void NormalizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	compute_norm<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels,height*width,bottom[0]->gpu_data(),norm_.mutable_gpu_data());
	
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height*width,bottom[0]->gpu_data(),norm_.gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void NormalizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels,height*width, top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_diff());
  
  backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_diff(),bottom[0]->mutable_gpu_diff());
}
template <typename Dtype>
void NormalizeLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels,height*width, bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_sec_diff());
  
  backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height*width,bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(NormalizeLayer);
}  // namespace caffe
