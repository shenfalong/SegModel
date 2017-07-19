
#include <vector>

#include "caffe/layers/activation/tanh_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void TanHForward(const int n, const Dtype* in, Dtype* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}

template <typename Dtype>
static __global__ void TanHBackward(const int n, const Dtype* in_diff,
    const Dtype* out_data, Dtype* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}
template <typename Dtype>
void TanHLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  TanHForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, bottom_data, top_data);
}

template <typename Dtype>
void TanHLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{

  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  TanHBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, top_diff, top_data, bottom_diff);
}
template <typename Dtype>
void TanHLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(TanHLayer);
}  // namespace caffe
