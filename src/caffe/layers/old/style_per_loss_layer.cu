
#include <vector>

#include "caffe/layers/style_per_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void gram_matrix_forward(int count, int channels,int height,int width, const Dtype *in, const Dtype *ref, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		Dtype sum = 0;
		for (int j = c;j < channels; j++)
		{
			Dtype delta = in[i]*in[((n*channels+j)*height+h)*width+w]/Dtype(channels) - ref[c*channels+j];
			sum += 0.5* delta*delta;
		}
		out[i] = sum / Dtype(channels);
	}
}
template <typename Dtype>
static __global__ void gram_matrix_backward(int count, int channels,int height,int width, const Dtype *in, const Dtype *ref, Dtype *diff_out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		Dtype sum = 0;
		
		for (int j = 0;j < channels; j++)
			sum += (in[i]*in[((n*channels+j)*height+h)*width+w]/Dtype(channels) - ref[c*channels+j])*in[((n*channels+j)*height+h)*width+w]/Dtype(channels);
		sum += (in[i]*in[i]/Dtype(channels) - ref[c*channels+c])*in[i]/Dtype(channels);
			
		diff_out[i] = sum / Dtype (channels);
	}
}
template <typename Dtype>
void StylePerLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	caffe_gpu_set(buffer_1_->count(),Dtype(0),buffer_1_->mutable_gpu_data());
	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels, channels, height*width,
														(Dtype)1., bottom[1]->gpu_data() + bottom[1]->offset(n), bottom[1]->gpu_data() + bottom[1]->offset(n), 
														(Dtype)1., buffer_1_->mutable_gpu_data());					
	}
	caffe_gpu_scal(buffer_1_->count(),Dtype(1)/Dtype(num*height*width*channels),buffer_1_->mutable_gpu_data());
	
	
	gram_matrix_forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height,width,bottom[0]->gpu_data(),buffer_1_->gpu_data(),buffer_square_->mutable_gpu_data());
	
	Dtype loss_sum;		
	caffe_gpu_asum(buffer_square_->count(),buffer_square_->gpu_data(),&loss_sum);											
	top[0]->mutable_cpu_data()[0] = loss_sum / Dtype(num*height*width*channels);
}

template <typename Dtype>
void StylePerLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	Dtype loss_weight_ = top[0]->cpu_diff()[0] / Dtype(num*height*width*channels);
	
	gram_matrix_backward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height,width,bottom[0]->gpu_data(),buffer_1_->gpu_data(),bottom[0]->mutable_gpu_diff());
	
	caffe_gpu_scal(bottom[0]->count(),loss_weight_,bottom[0]->mutable_gpu_diff());
}
INSTANTIATE_LAYER_GPU_FUNCS(StylePerLossLayer);
}  // namespace caffe
