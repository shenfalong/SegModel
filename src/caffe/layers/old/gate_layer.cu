
#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void gate_forward(int count,const Dtype * in_0, const Dtype * in_1, Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		out[i] = in_0[i] * in_1[i];		
	}
}

template <typename Dtype>
static __global__ void gate_backward(int count,const Dtype * diff_out,const Dtype *in_0, const Dtype *in_1, Dtype * diff_in_0, Dtype * diff_in_1)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		diff_in_0[i] = diff_out[i] * in_1[i];
		diff_in_1[i] = diff_out[i] * in_0[i];
	}
}
/*
template <typename Dtype>
__global__ void channels_reduction_forward(int count,int spatial_dim,const Dtype * in, Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int c = i / spatial_dim;
		int s = i % spatial_dim;
		
		out[i] = 0.25*in[(4*c+0)*spatial_dim+s] + 0.25*in[(4*c+1)*spatial_dim+s]
					 	+ 0.25*in[(4*c+2)*spatial_dim+s] + 0.25*in[(4*c+3)*spatial_dim+s];			 	
	}
}
template <typename Dtype>
__global__ void channels_reduction_backward(int count,int spatial_dim,const Dtype * diff_out, Dtype * diff_in)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int c = i / spatial_dim;
		int s = i % spatial_dim;
		
		diff_in[(4*c+0)*spatial_dim+s] = 0.25*diff_out[i];
		diff_in[(4*c+1)*spatial_dim+s] = 0.25*diff_out[i];
		diff_in[(4*c+2)*spatial_dim+s] = 0.25*diff_out[i];		
		diff_in[(4*c+3)*spatial_dim+s] = 0.25*diff_out[i]; 	
	}
}
*/
template <typename Dtype>
void GateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	//gate_forward<Dtype><<<CAFFE_GET_BLOCKS(buffer_top_->count()), CAFFE_CUDA_NUM_THREADS>>>
	//(buffer_top_->count(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),buffer_top_->mutable_gpu_data());	
	
	//channels_reduction_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	//(top[0]->count(),height*width,buffer_top_->gpu_data(),top[0]->mutable_gpu_data());
	
	gate_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());	
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

  //channels_reduction_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	//(top[0]->count(),height*width,top[0]->gpu_diff(),buffer_top_->mutable_gpu_diff());

	//gate_backward<Dtype><<<CAFFE_GET_BLOCKS(buffer_top_->count()), CAFFE_CUDA_NUM_THREADS>>>
	//(buffer_top_->count(), buffer_top_->gpu_diff(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),
	//							bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());		
	gate_backward<Dtype><<<CAFFE_GET_BLOCKS(buffer_top_->count()), CAFFE_CUDA_NUM_THREADS>>>
	(buffer_top_->count(), top[0]->gpu_diff(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),
								bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());		
}

INSTANTIATE_LAYER_GPU_FUNCS(GateLayer);
}  // namespace caffe
