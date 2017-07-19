#include <vector>

#include "caffe/layers/stuff_object_forward_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
static __global__ void max_stuff_kernel(int numspatial,int channels,int height,int width, const Dtype * in, const Dtype object_index ,Dtype * objectness, Dtype * out)
{
    CUDA_KERNEL_LOOP(i, numspatial)
    {
    	int n = i / width / height;
    	int h = i / width % height;
    	int w = i % width;
    	Dtype max_value = -100000;
    	int max_index = -1;
    	
    	for (int c=0;c<channels;c++)
    	{
    		int in_index = ((n*channels+c)*height+h)*width+w;
    		if (max_value < in[in_index])
    		{
    			max_value = in[in_index];
    			max_index = c;
    		}
    	}
    	
    	out[i] = max_index;
    	if (max_index == object_index)
    		objectness[i] = 1;
    	else
    		objectness[i] = 0;
    }
}    

template <typename Dtype>
static __global__ void max_object_kernel(int numspatial,int channels,int height,int width, const Dtype * in, Dtype * out)
{
    CUDA_KERNEL_LOOP(i, numspatial)
    {  		
	  	int n = i / width / height;
	  	int h = i / width % height;
	  	int w = i % width;
	  	
	  	int max_value = -100000;
	  	int max_index = -1;
	 
	  	for (int c=0;c<channels;c++)
	  	{
	  		int in_index = ((n*channels+c)*height+h)*width+w;
	  		if (max_value < in[in_index])
	  		{
	  			max_value = in[in_index];
	  			max_index = c;
	  		}
	  	}
	  	out[i] = max_index;	
    }
}    

template <typename Dtype>
static __global__ void stuff_object_mapping_kernel(int numspatial,int channels,int height,int width, const Dtype * in, const Dtype * objectness, const Dtype object_index , Dtype * out)
{
    CUDA_KERNEL_LOOP(i, numspatial)
    {
    	if (objectness[i] == object_index)
    	{  		
		  	int n = i / width / height;
		  	int h = i / width % height;
		  	int w = i % width;
		  	
		  	int max_value = -100000;
		  	int max_index = -1;
		 
		  	for (int c=0;c<channels;c++)
		  	{
		  		int in_index = ((n*channels+c)*height+h)*width+w;
		  		if (max_value < in[in_index])
		  		{
		  			max_value = in[in_index];
		  			max_index = c;
		  		}
		  	}
		
		  	out[i] = max_index;
		  }	
    }
}    

template <typename Dtype>
static __global__ void map_to_original_kernel(const int count,const Dtype *object_mapping, const Dtype * stuff_mapping, const Dtype * objectness,Dtype *out)
{
    CUDA_KERNEL_LOOP(i, count)
    {
    	if (objectness[i] == 1)		
		  	out[i] = object_mapping[int(out[i])];
		  else
		  	out[i] = stuff_mapping[int(out[i])];
    }
}

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  max_stuff_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width,channels,height,width,bottom[0]->gpu_data(),Dtype(35), objectness.mutable_gpu_data(),top[0]->mutable_gpu_data());
  
  max_object_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width,channels,height,width,bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  
  map_to_original_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (top[0]->count(),object_mapping.gpu_data(),stuff_mapping.gpu_data(),objectness.gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(StuffObjectForwardLayer);
}  // namespace caffe
		
