
#include <vector>

#include "caffe/layers/transfer_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define BN_EPS 1e-5

namespace caffe {
template <typename Dtype>
static __global__ void mymean_variance_forward(int channels, int spatial_dim, Dtype inv_norm_factor, const Dtype* bottom_data, Dtype* mean, Dtype* var) 
{
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

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
    mean[c] = buffer1[0] * inv_norm_factor;
     var[c] = buffer2[0] * inv_norm_factor;
  }
}

template <typename Dtype>
static __global__ void mykernel_test_forward(int count, int spatial_dim, const Dtype* scale, const Dtype* bias, const Dtype* mean, const Dtype* var, 
    Dtype eps, const Dtype* bottom_data,  Dtype* top_data) 
{
  CUDA_KERNEL_LOOP(i, count) 
  {
    int c = i / spatial_dim;
    top_data[i] = ((bottom_data[i] - mean[c]) / sqrt(var[c] + eps)) * sqrt(scale[c] + eps) + bias[c];
  }
}


template <typename Dtype>
static __global__ void mykernel_backward_scale_bias(int channels, int spatial_dim, const Dtype* mean, const Dtype* var, Dtype eps,
    const Dtype* top_diff, const Dtype* bottom_data,
    Dtype* scale_diff, Dtype* bias_diff) 
{
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < spatial_dim; i += blockDim.x) 
  {
    const int index = blockIdx.x * spatial_dim + i;
    buffer1[tid] += top_diff[index] * (bottom_data[index] - mean[c]) / sqrt(var[c] + eps);
    buffer2[tid] += top_diff[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) {
    scale_diff[c] = buffer1[0];
    bias_diff[c] = buffer2[0];
  }
}
template <typename Dtype>
static __global__ void mykernel_backward_bottom(int count, int spatial_dim, const Dtype* scale, const Dtype* bias,
    const Dtype* mean, const Dtype* var, Dtype eps, Dtype norm_factor, const Dtype* top_diff, const Dtype* scale_diff, const Dtype* bias_diff,
    const Dtype* bottom_data, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(i, count) 
  {
    int c = i / spatial_dim;
    const Dtype inv_std = Dtype(1) / sqrt(var[c] + eps);
    const Dtype x_norm = (bottom_data[i] - mean[c]) * inv_std;
    bottom_diff[i] = sqrt(scale[c]+eps) * inv_std * (top_diff[i] - (x_norm * scale_diff[c] + bias_diff[c]) / norm_factor);
    
    const Dtype inv_std_2 = Dtype(1) / sqrt(scale[c] + eps);
    const Dtype x_norm_2 = (bottom_data[i+count] - bias[c]) * inv_std_2;
    bottom_diff[i+count] = (bias_diff[c] + scale_diff[c]*x_norm_2) / norm_factor;
  }
}

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	mymean_variance_forward<<<num*channels, CAFFE_CUDA_NUM_THREADS>>>
	(num*channels, height*width, Dtype(1)/Dtype(height*width), bottom[0]->gpu_data(),mean_buffer_->mutable_gpu_data(), var_buffer_->mutable_gpu_data());
	
	caffe_gpu_mul(num*channels, mean_buffer_->gpu_data(), mean_buffer_->gpu_data(), mean_buffer_->mutable_gpu_diff());
	caffe_gpu_sub(num*channels, var_buffer_->gpu_data(), mean_buffer_->gpu_diff(), var_buffer_->mutable_gpu_data());
	
	mykernel_test_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()/2), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count()/2, height*width, 
	var_buffer_->gpu_data()+var_buffer_->count()/2, mean_buffer_->gpu_data()+mean_buffer_->count()/2, mean_buffer_->gpu_data(), var_buffer_->gpu_data(), 
	Dtype(BN_EPS), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 
 	mykernel_backward_scale_bias<<<num/2*channels, CAFFE_CUDA_NUM_THREADS>>>
 	(num/2*channels, height*width, mean_buffer_->gpu_data(), var_buffer_->gpu_data(),
	 Dtype(BN_EPS), top[0]->gpu_diff(), bottom[0]->gpu_data(),
	 var_buffer_->mutable_gpu_diff()+var_buffer_->count()/2,mean_buffer_->mutable_gpu_diff()+mean_buffer_->count()/2);
	               
	mykernel_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()/2),CAFFE_CUDA_NUM_THREADS>>>
	(	bottom[0]->count()/2, height*width, var_buffer_->gpu_data()+var_buffer_->count()/2, mean_buffer_->gpu_data()+mean_buffer_->count()/2, 
		mean_buffer_->gpu_data(), var_buffer_->gpu_data(), Dtype(BN_EPS), Dtype(height*width), 
		top[0]->gpu_diff(), var_buffer_->gpu_diff()+var_buffer_->count()/2, mean_buffer_->gpu_diff()+mean_buffer_->count()/2,
		bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(TransferBatchNormLayer);
}  // namespace caffe 
