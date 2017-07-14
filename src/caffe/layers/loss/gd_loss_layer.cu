#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss/gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void Dloss_forward_kernel(int count, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{		
		out[i] = max(in[i],Dtype(0))   + log(Dtype(1)+exp(-abs(in[i])))
					 + max(in[i+count],Dtype(0)) - in[i+count] + log(Dtype(1)+exp(-abs(in[i+count])));			
	}
}

template <typename Dtype>
static __global__ void Gloss_forward_kernel(int count, const Dtype *in, Dtype *out)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		out[i] =  max(in[i],Dtype(0))-in[i]+log(Dtype(1)+exp(-abs(in[i])));	
	}
}
template <typename Dtype>
static __global__ void Dloss_backward_kernel(int count, const Dtype *data_in, Dtype *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{	
		if (data_in[i] > 0) 
			diff_in[i] =  Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i])));
		else	
			diff_in[i] = 1 -  Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i])));
			
			
		if (data_in[i+count] > 0) 
			diff_in[i+count] = - 1 + Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i+count])));
		else
			diff_in[i+count] = - Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i+count])));
	}
}
template <typename Dtype>
static __global__ void Gloss_backward_kernel(int count, const Dtype *data_in, Dtype *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{		
		if (data_in[i] > 0) 
			diff_in[i] = -1 + Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i])));
		else
			diff_in[i] =  - Dtype(1.0) / (Dtype(1)+exp(-abs(data_in[i])));	
		diff_in[i+count] = 0;			
	}
}
//---------------------------------
template <typename Dtype>
static __global__ void Dloss_secforward_kernel(int count, const Dtype *in_sec_diff, const Dtype * in_data, Dtype *in_diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{	
		in_diff[i] =  in_sec_diff[i] * 	exp(-in_data[i]) / ( (Dtype(1)+exp(-in_data[i]))*(Dtype(1)+exp(-in_data[i])) );
		in_diff[i+count] = in_sec_diff[i+count] * exp(in_data[i+count]) / ( (Dtype(1)+exp(in_data[i+count]))*(Dtype(1)+exp(in_data[i+count])) );
	}
}
//---------------------------------
template <typename Dtype>
void GdLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	
	CHECK_EQ(bottom.size(),1);
	CHECK_EQ(num%2,0);
	CHECK_EQ(channels,1);

	if (Caffe::gan_type() == "train_dnet")
	{	
		Dloss_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*height*width,bottom[0]->gpu_data(),loss_.mutable_gpu_data());	
	}
	else
	{
		Gloss_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num/2*height*width,bottom[0]->gpu_data(),loss_.mutable_gpu_data());	
	}
	Dtype sum;
	caffe_gpu_asum(loss_.count(),loss_.gpu_data(),&sum);
	top[0]->mutable_cpu_data()[0] = sum / (num/2*height*width);
}

template <typename Dtype>
void GdLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	if (Caffe::second_pass() == false)
	{	
		if (Caffe::gan_type() == "train_dnet")
		{
			Dloss_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*height*width,bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());			
		}
		else
		{
			Gloss_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
			(num/2*height*width,bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());			
		}		
		Dtype loss_weights_ = top[0]->cpu_diff()[0] / (num/2*height*width);
		caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());		
	}
	else
	{	
	}
}
template <typename Dtype>
void GdLossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	Dloss_secforward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num/2*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num/2*height*width,bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());	
	
	Dtype loss_weights_ = top[0]->cpu_diff()[0] / (num/2*height*width);
	caffe_gpu_scal(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());	
}
INSTANTIATE_LAYER_GPU_FUNCS(GdLossLayer);
}  // namespace caffe
