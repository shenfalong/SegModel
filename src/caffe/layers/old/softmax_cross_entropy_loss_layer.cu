#include <algorithm>
#include <cfloat>
#include <vector>


#include "caffe/layers/softmax_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void SoftmaxCrossEntropyLossForwardGPU(const int nthreads,const int channels, const int spatial_dim,
          const Dtype* label_data,const Dtype* mask, const Dtype* prob_data,Dtype* loss) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {    
  	int n = index / spatial_dim / channels;
  	int s = index % spatial_dim;	
  	if (mask[n*spatial_dim + s] == 1.0)
    	loss[index] = -label_data[index] * log(max(prob_data[index], Dtype(FLT_MIN)));
    else
    	loss[index] = 0;
  }
}

template <typename Dtype>
static __global__ void SoftmaxCrossEntropyLossBackwardGPU(const int nthreads,const int channels, const int spatial_dim,
          const Dtype* label_data, const Dtype* mask, const Dtype* prob_data,Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
  	int n = index / spatial_dim / channels;
  	int c = index / spatial_dim % channels;
  	int s = index % spatial_dim;	
  	Dtype sum;	
  	if (mask[n*spatial_dim + s] == 1.0)
  	{
  		sum = 0;
			for (int cc=0;cc<channels;cc++)// please pay special attention to logical operation in CUDA !!!!
			{
				if (c==cc)
					sum += -label_data[(n*channels+cc)*spatial_dim+s]*(1.0 - prob_data[index]);
				else
					sum += -label_data[(n*channels+cc)*spatial_dim+s]*(0.0 - prob_data[index]);
			}
		} 
		else
		 sum = 0;
		bottom_diff[index] = sum;
  }
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

	const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  Dtype* loss_data = loss_.mutable_gpu_data(); 
 
  SoftmaxCrossEntropyLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels, height*width, label, bottom[2]->gpu_data(),prob_data,loss_data);
  
  Dtype loss, count;
  caffe_gpu_asum(loss_.count(), loss_data, &loss);
  caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &count);
  
	if (count > 0)
  	top[0]->mutable_cpu_data()[0] = loss / count;
  else
  	top[0]->mutable_cpu_data()[0] = 0; 	 	
}

template <typename Dtype>
void SoftmaxCrossEntropyLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	 Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 

  SoftmaxCrossEntropyLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels, height*width, label, bottom[2]->gpu_data(),prob_data,bottom[0]->mutable_gpu_diff());

  Dtype count;
  caffe_gpu_asum(bottom[2]->count(), bottom[2]->gpu_data(), &count);
  
  Dtype loss_weight;
  if (count > 0)
  	loss_weight = top[0]->cpu_diff()[0] / count;
  else
  	loss_weight = 0;
  	
  caffe_gpu_scal(bottom[0]->count(), loss_weight , bottom_diff);
}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxCrossEntropyLossLayer);
}  // namespace caffe
