
#include <vector>

#include "caffe/layers/operator/mean_variance_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define BN_EPS 1e-5
namespace caffe {

template <typename Dtype>
static __global__ void mean_variance_forward(int channels,int spatial_dim, Dtype inv_norm_factor, const Dtype* bottom_data, Dtype* top_data) 
{
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x % channels;
  const int n = blockIdx.x / channels;


  buffer1[tid] = 0;
  buffer2[tid] = 0;
  for (int i = tid; i < spatial_dim; i += blockDim.x) 
  {
    const int index = blockIdx.x * spatial_dim + i;
    buffer1[tid] += bottom_data[index];
    buffer2[tid] += bottom_data[index] * bottom_data[index];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) 
  {
    top_data[n*2*channels+         c] = buffer1[0] * inv_norm_factor;
    top_data[n*2*channels+channels+c] = buffer2[0] * inv_norm_factor;
  }
}
template <typename Dtype>
static __global__ void square_root(int count, int channels,const Dtype * mean_square, Dtype * mean_var)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	const int c = i % channels;
  	const int n = i / channels;
  
  	mean_var[n*2*channels+c] = mean_square[n*2*channels+c];
  	
		mean_var[n*2*channels+channels+c] = sqrt(mean_square[n*2*channels+channels+c] - mean_square[n*2*channels+c]*mean_square[n*2*channels+c] + BN_EPS);
	}
}

template <typename Dtype>
static __global__ void mean_variance_backward(int count, int channels, int spatial_dim, Dtype inv_norm_factor,
    const Dtype* top_diff, const Dtype* top_data, const Dtype* bottom_data, Dtype* bottom_diff) 
{
	CUDA_KERNEL_LOOP(i, count)
  {
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		
		bottom_diff[i] = top_diff[n*2*channels+c]*inv_norm_factor + 
										 top_diff[n*2*channels+channels+c]*inv_norm_factor*(bottom_data[i]-top_data[n*2*channels+c])/top_data[n*2*channels+channels+c];
	}
}
template <typename Dtype>
void MeanVarianceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	mean_variance_forward<<<num*channels, CAFFE_CUDA_NUM_THREADS>>>
	(channels, height*width, Dtype(1)/Dtype(height*width), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
	
	square_root<<<CAFFE_GET_BLOCKS(num*channels), CAFFE_CUDA_NUM_THREADS>>>
	(num*channels,channels,top[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void MeanVarianceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 
	mean_variance_backward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), channels, height * width, Dtype(1)/Dtype(height*width),top[0]->gpu_diff(), top[0]->gpu_data(),bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
}
template <typename Dtype>
void MeanVarianceLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(MeanVarianceLayer);
}  // namespace caffe
