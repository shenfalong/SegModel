
#include <vector>

#include "caffe/layers/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void kernel(int count, int channels,int height,int width, const Dtype *in, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		out[i] = in[i];
	}
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height,width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	
}

INSTANTIATE_LAYER_GPU_FUNCS(ChannelReductionLayer);
}  // namespace caffe
