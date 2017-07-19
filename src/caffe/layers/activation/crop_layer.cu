#include <vector>

#include "caffe/layers/activation/crop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void crop_forward_kernel(int count,int channels,int height,int width,int pad, const Dtype * in, Dtype * out)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height / channels;
  	int c = index / width / height % channels;
  	int h = index / width % height;
  	int w = index % width;
  	 	
  	out[index] = in[((n*channels+c)*(height+2*pad)+pad+h)*(width+2*pad)+pad+w];
  }
}    

template <typename Dtype>
static __global__ void crop_backward_kernel(int count,int channels,int height,int width,int pad, const Dtype * out_diff, Dtype * in_diff)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height / channels;
  	int c = index / width / height % channels;
  	int h = index / width % height;
  	int w = index % width;
  	  	
  	in_diff[((n*channels+c)*(height+2*pad)+pad+h)*(width+2*pad)+pad+w] = out_diff[index];
  }
} 

template <typename Dtype>
void CropLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int channels = top[0]->channels();
	int height = top[0]->height();
  int width = top[0]->width();
  
  crop_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (top[0]->count(),channels,height,width,pad_,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void CropLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
	
	int channels = top[0]->channels();
	int height = top[0]->height();
  int width = top[0]->width();

	crop_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (top[0]->count(),channels,height,width,pad_,top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
}
template <typename Dtype>
void CropLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}
INSTANTIATE_LAYER_GPU_FUNCS(CropLayer);
}  // namespace caffe
		
