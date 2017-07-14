#include <vector>

#include "caffe/layers/condition_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
static __global__ void condition_filter_forward_kernel(int count, int channels,int height, int width, const Dtype * in_0, const Dtype *in_1, Dtype * out)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height;
  	int h = index / width % height;
  	int w = index % width;
  	int cur_channel = in_1[index];
  	if (cur_channel == 255)
  		 cur_channel = 0;
  	for(int c =0;c<channels;c++)
  	{
  		int out_ind  = ((n*channels+c)*height+h)*width+w;
  		
  		int in_0_ind  = (((n*channels+cur_channel)*channels+c)*height+h)*width+w;
  		out[out_ind] = in_0[in_0_ind];  
  	}
  }
}    

template <typename Dtype>
static __global__ void condition_filter_backward_kernel(int count, int channels,int height, int width, const Dtype * out_diff, const Dtype *in_1, Dtype * in_0_diff)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height;
  	int h = index / width % height;
  	int w = index % width;
  	int cur_channel = in_1[index];
  	if (cur_channel == 255)
  		 cur_channel = 0;
  	for(int c =0;c<channels;c++)
  	{
  		int out_ind  = ((n*channels+c)*height+h)*width+w;
  		
  		int in_0_ind  = (((n*channels+cur_channel)*channels+c)*height+h)*width+w;
  		in_0_diff[in_0_ind] = out_diff[out_ind];  
  	}
  }
}    

template <typename Dtype>
void ConditionFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = top[0]->num();
	int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
  condition_filter_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
   (num*height*width,channels,height,width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void ConditionFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());

	int num = top[0]->num();
	int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
	
	condition_filter_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width,channels,height,width,top[0]->gpu_diff(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(ConditionFilterLayer);
}  // namespace caffe
		
