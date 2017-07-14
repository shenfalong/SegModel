#include <vector>

#include "caffe/layers/mask_out_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void ProdForward_kernel(const int count, const int channels, const int spatial,const Dtype * in_0, const Dtype * ind_1, Dtype  *out)
{
	CUDA_KERNEL_LOOP(ind, count) 
  { 
  	int n = ind / spatial / channels;
  	//int c = ind / spatial % channels;
  	int s = ind % spatial;
  	int mask_ind = n * spatial + s;
  	out[ind] = in_0[ind]*ind_1[mask_ind];
  }
}  

template <typename Dtype>
static __global__ void ProdBackward_kernel(const int count, const int channels, const int spatial,const Dtype * out_diff,const Dtype * ind_1, Dtype * ind_0_diff)
{
	CUDA_KERNEL_LOOP(ind, count) 
  { 
  	int n = ind / spatial / channels;
  	//int c = ind / spatial % channels;
  	int s = ind % spatial;
  	int mask_ind = n * spatial + s;
  	ind_0_diff[ind] = out_diff[ind]*ind_1[mask_ind];
  }
}

template <typename Dtype>
void MaskOutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	ProdForward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void MaskOutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	ProdBackward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), channels,height*width,top[0]->gpu_diff(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(MaskOutLayer);
}  // namespace caffe
