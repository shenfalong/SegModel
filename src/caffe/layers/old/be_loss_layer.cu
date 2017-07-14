
#include <vector>

#include "caffe/layers/be_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
#if 0
template <typename Dtype>
static__global__ void dnet_loss_kernel(int count,  const Dtype *reconstruct, const Dtype *input, Dtype *real_loss, Dtype *fake_loss)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		real_loss[i] = abs(input[i] - reconstruc[i]);
		fake_loss[i] = abs(input[i+count] - reconstruct[i+count]);
	}
}

template <typename Dtype>
static __global__ void dnet_gradient_kernel(int count,  const Dtype *reconstruct, const Dtype *input, Dtype k, Dtype *diff_reconstruct)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		if (input[i] > reconstruc[i])
			diff_reconstruc[i] = Dtype(-1);
		else
			diff_reconstruc[i] = Dtype(1);
		
		if (input[i] > reconstruc[i])
			diff_reconstruc[i+count] = k;
		else
			diff_reconstruc[i+count] = -k;				
	}
}
#endif
template <typename Dtype>
void BeLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
#if 0
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	CHECK_EQ(num%2,0);
	
	dnet_loss_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()/2), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count()/2,bottom[0]->gpu_data(),bottom[1]->gpu_data(),real_loss_.mutable_gpu_data(),fake_loss_.mutable_gpu_data());
	Dtype real_loss, fake_loss;
	caffe_gpu_asum(real_loss_.count(),real_loss_.gpu_data(),&real_loss);
	caffe_gpu_asum(fake_loss_.count(),fake_loss_.gpu_data(),&fake_loss);
	
	
	Dtype loss = real_loss / Dtype(real_loss_.count()) - k_ * fake_loss / Dtype(fake_loss_.count());
	top[0]->mutable_gpu_data()[0] = loss;
	
	k_ = 0.5;
	//k_ = k_ + 0.001*(0.3*real_loss - fake_loss);
#endif	
}
 
template <typename Dtype>
void BeLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
#if 0
	dnet_gradient_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()/2), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count()/2,bottom[0]->gpu_data(),bottom[1]->gpu_data(),k_,bottom[0]->mutable_gpu_diff());
#endif
}

INSTANTIATE_LAYER_GPU_FUNCS(BeLossLayer);
}  // namespace caffe
