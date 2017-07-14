#include <vector>

#include "caffe/layers/project_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void per_project(const int count,const Dtype * in,Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		if (in[i] == 255)
			out[i]=255;
		else
			out[i]=in[i]>0;
			
	}

}

template <typename Dtype>
static __global__ void max_project(const int count, const int channels, const int height, const int width, const Dtype * in,Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i/height/width;
		int h = (i/width)%height;
		int w = i % width;
		int out_index = ((n*2+1)*height+h)*width+w;
		
		Dtype max_value = -100;
		int max_index = 0;
		for (int j=1;j<channels;j++)
		{
			int in_index = ((n*channels+j)*height+h)*width+w;
			if (max_value < in[in_index])
			{
				max_value = in[in_index];
				max_index = in_index;
			}
		}
		out[out_index] = in[max_index];
			
	}

}

template <typename Dtype>
void ProjectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	if (channels == 21)
	{	
		for (int n=0;n<num;n++)	
			caffe_copy(height*width,bottom[0]->gpu_data()+n*channels*height*width,top[0]->mutable_gpu_data()+n*2*height*width);
			
		max_project<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num*height*width,channels,height,width, bottom[0]->gpu_data(),top[0]->mutable_gpu_data());	
	}
	else if (channels == 1)
	{
		per_project<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(bottom[0]->count(), bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	}		
}

template <typename Dtype>
void ProjectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	if (bottom[0]->channels() == 21)
	{
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ProjectLayer);
}  // namespace caffe
