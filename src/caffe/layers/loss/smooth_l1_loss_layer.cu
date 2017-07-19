#include "caffe/layers/loss/smooth_l1_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void SmoothL1Forward(const int n, const int ignore_value, const Dtype* in_0, const Dtype * in_1, Dtype* out, Dtype * count) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	if (in_0[index] == ignore_value)
  	{
  		count[index] = 0;
  		out[index] = 0;
  	}
  	else
  	{
			count[index] = 1;
			
		  Dtype val = abs(in_0[index] - in_1[index]);
			
		  out[index] = val;
		}    
  }
}
template <typename Dtype>
static __global__ void SmoothL1Backward(const int n, const int ignore_value, const Dtype* in_0, const Dtype * in_1, Dtype* in_0_diff, Dtype * count) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	if (in_0[index] == ignore_value)
  	{
  		count[index] = 0;
  		in_0_diff[index] = 0;
  	}
  	else
  	{
		  Dtype val = in_0[index] - in_1[index];
		  if (val > 0)
		  	in_0_diff[index] = Dtype(1);
		  else
		  	in_0_diff[index] = Dtype(-1);
		}    
  }
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  
  SmoothL1Forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),ignore_value,bottom[0]->gpu_data(),bottom[1]->gpu_data(), loss_.mutable_gpu_data(),counts_.mutable_gpu_data());

	Dtype counts;
  caffe_gpu_asum(counts_.count(), counts_.gpu_data(), &counts);
  
  Dtype loss;
  caffe_gpu_asum(loss_.count(), loss_.gpu_data(), &loss);
  
  top[0]->mutable_cpu_data()[0] = loss / counts;
}

template <typename Dtype>
void SmoothL1LossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
  SmoothL1Backward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),ignore_value,bottom[0]->gpu_data(),bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff(),counts_.mutable_gpu_data());
  
  Dtype counts;
  caffe_gpu_asum(counts_.count(), counts_.gpu_data(), &counts);
  
  caffe_gpu_scal(bottom[0]->count(),top[0]->cpu_diff()[0] / counts,bottom[0]->mutable_gpu_diff());
}
template <typename Dtype>
void SmoothL1LossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(SmoothL1LossLayer);

}  // namespace caffe
