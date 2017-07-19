#include <vector>

#include "caffe/layers/activation/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
static __global__ void DropoutForward(const int count, int channels, int spatial, const Dtype threshold, const Dtype* in, const Dtype* rand_vec, Dtype* out) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	int c = index / spatial % channels;
  	if (rand_vec[c] > threshold)
    	out[index] = in[index];
    else
    	out[index] = 0;
  }
}

template <typename Dtype>
static __global__ void DropoutBackward(const int count, int channels, int spatial, const Dtype threshold, const Dtype* in_diff, const Dtype * rand_vec, Dtype* out_diff) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	int c = index / spatial % channels;
  	if (rand_vec[c] > threshold)
    	out_diff[index] = in_diff[index];
   	else
    	out_diff[index] = 0;
  }
}

template <typename Dtype>
void DropoutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();	
	
  if (Caffe::drop_state() == "rand")
  {
    caffe_gpu_rng_uniform(channels,Dtype(0.0),Dtype(1.0), rand_vec_.mutable_gpu_data());
    

    DropoutForward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
    (bottom[0]->count(),channels,height*width,threshold_, bottom[0]->gpu_data(), rand_vec_.gpu_data(), top[0]->mutable_gpu_data());
  } 
  else 
  {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
    caffe_gpu_scal(top[0]->count(),Dtype(1.0)-threshold_,top[0]->mutable_gpu_data());
  }
  
}

template <typename Dtype>
void DropoutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();	
  
  
  DropoutBackward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels,height*width, threshold_, top[0]->gpu_diff(), rand_vec_.gpu_data(), bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
void DropoutLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
}

INSTANTIATE_LAYER_GPU_FUNCS(DropoutLayer);

}  // namespace caffe
