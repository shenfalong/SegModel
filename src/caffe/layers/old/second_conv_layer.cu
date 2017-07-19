
#include <vector>

#include "caffe/layers/second_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//-------------------------------------------------------------------
template <typename Dtype>
static __global__ void inner_matrix_multiply_forward(int count, int channels, int channels_x, int height, int width, 
																					const Dtype * in0, const Dtype * in1, Dtype * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int nw = i / width;
  	int nh = nw / height;
  	int n = nh / channels;

  	int w = i - nw * width;
  	int h = nw - nh * height;
  	int c = nh - n * channels;
  	
  	float sum = 0;
		for (int k = 0;k < channels_x; k++)
			sum += in0[(((n*channels+c)*channels_x+k)*height+h)*width+w] * in1[((n*channels_x+k)*height+h)*width+w];

		out[i] = sum;
	}
}
template <typename Dtype>
static __global__ void inner_matrix_multiply_backward(int count, int channels, int channels_x, int height, int width, 
																			const Dtype * diff_out, const Dtype * in0, const Dtype * in1, Dtype * diff_in0, Dtype * diff_in1)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int nw = i / width;
  	int nh = nw / height;
  	int n = nh / channels_x;

  	int w = i - nw * width;
  	int h = nw - nh * height;
  	int c = nh - n * channels_x;
  	
  	float sum = 0;
		for (int k=0; k<channels; k++)
		{	
			diff_in0[(((n*channels+k)*channels_x+c)*height+h)*width+w] 
											= diff_out[((n*channels+k)*height+h)*width+w] * in1[((n*channels_x+c)*height+h)*width+w];
			
			sum += in0[(((n*channels+k)*channels_x+c)*height+h)*width+w] * diff_out[((n*channels+k)*height+h)*width+w];
		}
		diff_in1[i]= sum;
	}
}
//-------------------------------------------------------------------

template <typename Dtype>
void SecConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels_sec = bottom[0]->channels();
  int channels_x = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	inner_matrix_multiply_forward<Dtype><<<CAFFE_GET_BLOCKS(num*channels_sec/channels_x*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*channels_sec/channels_x*height*width, channels_sec/channels_x, channels_x, height, width, 
  		bottom[0]->gpu_data(),  bottom[1]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
void SecConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels_sec = bottom[0]->channels();
  int channels_x = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	inner_matrix_multiply_backward<Dtype><<<CAFFE_GET_BLOCKS(bottom[1]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[1]->count(), channels_sec/channels_x, channels_x,  height, width, 
			top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SecConvLayer);
}  // namespace caffe
