
#include <vector>

#include "caffe/layers/data/interp_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void forward_kernel(int count, int image_dim, const Dtype *in_0, const Dtype *in_1, const Dtype *coef, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / image_dim;
		out[i] = coef[n]*in_0[i] + (1-coef[n])*in_1[i];
	}
}


template <typename Dtype>
void InterpDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	CHECK_EQ(bottom[0]->count(),bottom[1]->count());

	coef_.Reshape(num,1,1,1);
	caffe_rng_uniform<Dtype>(coef_.count(),Dtype(0), Dtype(1), coef_.mutable_cpu_data());
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels*height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(), coef_.gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void InterpDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
}
template <typename Dtype>
void InterpDataLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(InterpDataLayer);
}  // namespace caffe
