#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/stuff_object_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void stuff_object_kernel_forward(const int count,const int channels_0, const int channels_1, const int spatial,const Dtype *stuff_mapping, const Dtype *object_mapping,
																																																			const Dtype *in_0, const Dtype *in_1, Dtype *out)
{
  CUDA_KERNEL_LOOP(i, count)
  {
  	int n = i / spatial / (channels_0 - 1 + channels_1);
  	int c = i / spatial % (channels_0 - 1 + channels_1);
  	int s = i %  spatial;
  	if (c < channels_0 -1)
  	{
  		int in_index = (n * channels_0 + c)*spatial + s;
  		out[i] = in_0[in_index];
  	}	
  	else
  	{
  		int in_index_0 = (n * channels_0 + channels_0-1)*spatial + s;
  		int in_index_1 = (n * channels_1 + c-(channels_0-1))*spatial + s;
  		out[i] = in_0[in_index_0] + log(max(in_1[in_index_1],Dtype(FLT_MIN)));
  	}	 
  }
}    

template <typename Dtype>
static __global__ void map_original_kernel(const int count,const int channels_0, const int channels_1, const int spatial,const Dtype * stuff_mapping, const Dtype *object_mapping, 
																																						const Dtype * in, Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial / (channels_0 - 1 + channels_1);
		int c = i / spatial % (channels_0 - 1 + channels_1);
		int s = i %  spatial;
		if (c < channels_0 - 1)
		{
			int out_index = (n * (channels_0 - 1 + channels_1) + int(stuff_mapping[c]))*spatial + s;
		 	out[out_index] = in[i];
		
		}
		else
		{
			int out_index = (n * (channels_0 - 1 + channels_1) + int(object_mapping[c-(channels_0-1)]))*spatial + s;
			out[out_index] = in[i];
		}
	}
}
//----------------------------------------------------------------------------------------------------------------
template <typename Dtype>
static __global__ void stuff_object_kernel_backward(const int count,const int channels_0, const int channels_1, const int spatial, const Dtype *out_diff, const Dtype * in_1_data,
																																																				Dtype *in_0_diff, Dtype *in_1_diff)
{
  CUDA_KERNEL_LOOP(i, count)
  {
  	int n = i / spatial / (channels_0 - 1 + channels_1);
  	int c = i / spatial % (channels_0 - 1 + channels_1);
  	int s = i %  spatial;
  	if (c < channels_0-1)
  	{
  		int in_index = (n * channels_0 + c)*spatial + s;
  		in_0_diff[in_index] = out_diff[i];
  	}	
  	else
  	{
  		int in_index = (n * channels_1 + c-(channels_0-1))*spatial + s;
  		in_1_diff[in_index] = out_diff[i] * 1/(max(in_1_data[in_index],Dtype(FLT_MIN)));
  	}	 
  }
}    

template <typename Dtype>
static __global__ void original_map_kernel(const int count,const int channels_0, const int channels_1, const int spatial, const Dtype * stuff_mapping, const Dtype *object_mapping,
																																						const Dtype * out_diff, Dtype * in_diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial / (channels_0 - 1 + channels_1);
		int c = i / spatial % (channels_0 - 1 + channels_1);
		int s = i %  spatial;
		if (c < (channels_0-1))
		{
			int out_index = (n * (channels_0 - 1 + channels_1) + int(stuff_mapping[c]))*spatial + s;
		 	in_diff[i] = out_diff[out_index];
		
		}
		else
		{
			int out_index = (n * (channels_0 - 1 + channels_1) + int(object_mapping[c-(channels_0-1)]))*spatial + s;
			in_diff[i] = out_diff[out_index];
		}
	}
}

template <typename Dtype>
static __global__ void gradient_collect(const int numspatial,const int channels_0, const int channels_1, const int spatial, const Dtype * out_diff, Dtype * in_diff)
{
	CUDA_KERNEL_LOOP(i, numspatial)
	{
		int n = i / spatial;
		int s = i %  spatial;
		int in_index = (n * channels_0 + channels_0 - 1)*spatial + s;
		Dtype sum = 0;
		for (int c=0;c<channels_1;c++)
		{
			int out_index = (n * (channels_0 - 1 + channels_1) + channels_0 - 1 + c)*spatial + s;
			sum += out_diff[out_index];
		}
		in_diff[in_index] = sum;
	}
}
//-----------------------------------------------------------------------------------------------------


template <typename Dtype>
void StuffObjectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels_0 = bottom[0]->channels();
	int channels_1 = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  
  stuff_object_kernel_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (top[0]->count(),channels_0,channels_1,height*width,stuff_mapping.gpu_data(),object_mapping.gpu_data(),
  bottom[0]->gpu_data(),prob_.gpu_data(),reorder_top.mutable_gpu_data());
  
  
  map_original_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (top[0]->count(),channels_0,channels_1,height*width,stuff_mapping.gpu_data(),object_mapping.gpu_data(),
  reorder_top.gpu_data(), top[0]->mutable_gpu_data()); 
}

template <typename Dtype>
void StuffObjectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{	
	int num = bottom[0]->num();
	int channels_0 = bottom[0]->channels();
	int channels_1 = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
  
	original_map_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels_0,channels_1,height*width,stuff_mapping.gpu_data(),object_mapping.gpu_data(),
  top[0]->gpu_diff(),reorder_top.mutable_gpu_diff());

	stuff_object_kernel_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels_0,channels_1,height*width, reorder_top.gpu_diff(),prob_.gpu_data(),
	bottom[0]->mutable_gpu_diff(),prob_.mutable_gpu_diff());
  
	
	gradient_collect<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels_0,channels_1,height*width, reorder_top.gpu_diff(),bottom[0]->mutable_gpu_diff());
	
	softmax_layer_->Backward(softmax_top_vec_,softmax_bottom_vec_);
}

INSTANTIATE_LAYER_GPU_FUNCS(StuffObjectLayer);
}  // namespace caffe
		
