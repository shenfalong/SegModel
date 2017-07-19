#include <algorithm>
#include <vector>

#include "caffe/layers/activation/elu_layer.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void forward_kernel(const int n, const Dtype* in, Dtype* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    out[index] = in[index] > 0 ? in[index] : exp(in[index]) - 1;
  }
}

template <typename Dtype>
static __global__ void backward_kernel_0(const int n, const Dtype* out_diff, const Dtype* in_data, Dtype* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] = in_data[index] > 0 ? out_diff[index] :out_diff[index] * exp(in_data[index]);
  }
}
template <typename Dtype>
static __global__ void backward_kernel_1(const int n, const Dtype* out_diff, const Dtype* in_data, Dtype* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] += in_data[index] > 0 ? out_diff[index] :out_diff[index] * exp(in_data[index]);
  }
}
template <typename Dtype>
static __global__ void secforward_kernel_diff(const int n, const Dtype* in_sec_diff, const Dtype* in_data, Dtype* out_sec_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    out_sec_diff[index] = in_data[index] > 0 ? in_sec_diff[index] :in_sec_diff[index] * exp(in_data[index]);
  }
}
template <typename Dtype>
static __global__ void secforward_kernel_data(const int n, const Dtype* in_sec_diff, const Dtype * out_diff, const Dtype* in_data, Dtype* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] = in_data[index] > 0 ? 0 :in_sec_diff[index] * out_diff[index] * exp(in_data[index]);
  }
}
template <typename Dtype>
void ELULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void ELULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	if (this->has_bottom_sec_diff_ ==  false)
	{
		backward_kernel_0<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
	}
	else
	{
		backward_kernel_1<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
		this->has_bottom_sec_diff_ = false;
	}	
}

template <typename Dtype>
void ELULayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

	secforward_kernel_diff<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_sec_diff(),  bottom[0]->gpu_data(), top[0]->mutable_gpu_sec_diff());
	
	secforward_kernel_data<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_sec_diff(), top[0]->gpu_diff(),  bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
	this->has_bottom_sec_diff_ = true;
}
INSTANTIATE_LAYER_GPU_FUNCS(ELULayer);

}  // namespace caffe

