#include <vector>

#include "caffe/layers/shortcut_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void short_forward_kernel(int count, int channels, int spatial_dim,const Dtype * in_0, const Dtype * weight,const Dtype * in_1, Dtype * out)
{
    CUDA_KERNEL_LOOP(n, count)
    {
    	int c = n / spatial_dim % channels;
			out[n] = in_0[n]  + in_1[n] * weight[c];
    }
} 

template <typename Dtype>
static __global__ void short_backward_kernel(int count, int channels, int spatial_dim, const Dtype * weight,const Dtype * out_diff, Dtype * in_1_diff)
{
    CUDA_KERNEL_LOOP(n, count)
    {
    	int c = n / spatial_dim % channels;
			in_1_diff[n] = out_diff[n] * weight[c];		
    }
} 


template <typename Dtype>
void ShortcutChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	short_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels,height*width,bottom[0]->gpu_data(),this->blobs_[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
				
}

template <typename Dtype>
void ShortcutChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	
	caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	
//---------------------------------------------------------------
	lambda->Reshape(num, channels, height, width);
	all_one->Reshape(1, 1, height, width);
  caffe_gpu_set(all_one->count(),Dtype(1.0),all_one->mutable_gpu_data());
//---------------------------------------------------------------
	
	short_backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels,height*width,this->blobs_[0]->gpu_data(),top[0]->gpu_diff(),bottom[1]->mutable_gpu_diff());
  
  
  caffe_gpu_mul(top[0]->count(),top[0]->gpu_diff(),bottom[1]->gpu_data(),lambda->mutable_gpu_data());
  for (int i=0;i<num;i++)
  {
		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels,1, height*width, 
												(Dtype)1., lambda->gpu_data() + i * channels * height * width, all_one->gpu_data(),
												(Dtype)1., this->blobs_[0]->mutable_gpu_diff());
	}											
				
}

INSTANTIATE_LAYER_GPU_FUNCS(ShortcutChannelLayer);
}  // namespace caffe
