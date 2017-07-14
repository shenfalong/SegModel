#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/entropy_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void entropymask_kernel(const int numspatial,const int num, const int channels, const int spatial,const Dtype * in, Dtype  *out)
{
	CUDA_KERNEL_LOOP(ind, numspatial) 
  { 
  	int n = ind / spatial;
  	int s = ind % spatial;
  	Dtype sum = 0;
  	for (int c=0;c<channels;c++)
			sum += -in[(n * channels + c) * spatial+s]*log(max(in[(n * channels + c) * spatial+s], Dtype(FLT_MIN)));
		out[ind] = sum;
	}
}

template <typename Dtype>
__global__ void diff_kernel(const int numspatial,const int num, const int height,const int width,const Dtype * in, Dtype  *out)
{
	CUDA_KERNEL_LOOP(ind, numspatial) 
  { 
  	int n = ind / width / height;
  	int h = ind / width % height;
  	int w = ind % width;
  	int in_ind_x = (n * height + h) * width + max(w - 1,0); 
  	int in_ind_y = (n * height + max(h-1,0)) * width + w; 
		Dtype diff = sqrt((in[ind] - in[in_ind_x])*(in[ind] - in[in_ind_x]) + (in[ind] - in[in_ind_y])*(in[ind] - in[in_ind_y]));
		out [ind] = diff > 0.1;
		
	}
}
template <typename Dtype>
void EntropyMaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  Dtype * mask_data = top[0]->mutable_gpu_data();
  
	softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
	const Dtype* prob_data = prob_.gpu_data();
	 
	entropymask_kernel<Dtype><<<CAFFE_GET_BLOCKS(num * height * width), CAFFE_CUDA_NUM_THREADS>>>
  (num * height * width, num, channels, height*width, prob_data, mask_data);
  
  diff_kernel<Dtype><<<CAFFE_GET_BLOCKS(num * height * width), CAFFE_CUDA_NUM_THREADS>>>
  (num * height * width, num, height,width,mask_data, mask_data);
}

template <typename Dtype>
void EntropyMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	//do nothing
}

INSTANTIATE_LAYER_GPU_FUNCS(EntropyMaskLayer);
}  // namespace caffe
