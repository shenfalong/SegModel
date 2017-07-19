#include <vector>

#include "caffe/layers/same_max_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
static __global__ void same_max_forward_kernel(int numspatial,int channels,int spatial,const Dtype * in_0, const Dtype *in_1,Dtype* out_index, Dtype * out)
{
    CUDA_KERNEL_LOOP(index, numspatial)
    {
    	int n = index / spatial;
    	int s = index % spatial;
    	Dtype max_value_0 = - 1000;
    	Dtype max_value_1 = - 1000;
    	Dtype max_index_0 = - 1;
    	Dtype max_index_1 = - 1;
    	for (int c = 0; c < channels; c++)
    	{
    	  int cur_index = (n * channels + c)*spatial+s;
    		if(in_0[cur_index] > max_value_0)
    		{
    			max_value_0 = in_0[cur_index];
    			max_index_0 = c;
    		}
    		if(in_1[cur_index] > max_value_1)
    		{
    			max_value_1 = in_1[cur_index];
    			max_index_1 = c;
    		}
    	}
    	if (max_index_0 == max_index_1 || max_value_0 >= max_value_1)
    	{
    		out_index[index] = 0;
    		for (int c = 0; c < channels; c++)
    		{
    			int cur_index = (n * channels + c)*spatial+s;
    			out[cur_index] = in_0[cur_index];
    		}
    	}
    	else
    	{
    		out_index[index] = 1;
    		for (int c = 0; c < channels; c++)
    		{
    			int cur_index = (n * channels + c)*spatial+s;
    			out[cur_index] = in_1[cur_index];
    		}
    	}
    	
    }
}    
template <typename Dtype>
static __global__ void same_max_backward_kernel(int numspatial,int channels,int spatial, const Dtype* out_index, const Dtype * out_diff, Dtype * in_0_diff, Dtype *in_1_diff)
{
    CUDA_KERNEL_LOOP(index, numspatial)
    {
    	int n = index / spatial;
    	int s = index % spatial;
    	if (out_index[index] == 0)
    	{
    		for (int c = 0; c < channels; c++)
    		{
    			int cur_index = (n * channels + c)*spatial+s;
    			in_0_diff[cur_index] = out_diff[cur_index];
    		}
    	}
    	else
    	{
    		for (int c = 0; c < channels; c++)
    		{
    			int cur_index = (n * channels + c)*spatial+s;
    			in_1_diff[cur_index] = out_diff[cur_index];
    		}	
    	}
    }
}    

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  same_max_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width,channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),mask.mutable_gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
	caffe_gpu_set(bottom[1]->count(),Dtype(0),bottom[1]->mutable_gpu_diff());
		
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	same_max_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width,channels,height*width,mask.gpu_data(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());
  
  
}

INSTANTIATE_LAYER_GPU_FUNCS(SameMaxPoolingLayer);
}  // namespace caffe
		
