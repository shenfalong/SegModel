
#include <vector>

#include "caffe/layers/loss/max_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void forward_kernel(int count, int channels,int spatial_dim, const Dtype *pred_data, const Dtype *label_data, Dtype *loss_data)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		int label = label_data[i];
		Dtype sum = 0;
		for (int c=0;c<channels;c++)
		{
			if (c != label)
				sum += abs(pred_data[(n*channels+c)*spatial_dim+s]);
			else
				sum += abs(1 - pred_data[(n*channels+c)*spatial_dim+s]);
		}
		loss_data[i] = sum;
	}
}
template <typename Dtype>
static __global__ void backward_kernel(int count, int channels,int spatial_dim, const Dtype *pred_data, const Dtype *label_data, Dtype *pred_diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		int label = label_data[i];
		for (int c=0;c<channels;c++)
		{
			if (c != label)
				pred_diff[(n*channels+c)*spatial_dim+s] = pred_data[(n*channels+c)*spatial_dim+s]>0? 1:-1;
			else
				pred_diff[(n*channels+c)*spatial_dim+s] = pred_data[(n*channels+c)*spatial_dim+s]-1>0? 1:-1;
		}	
	}
}
template <typename Dtype>
void MaxLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),loss_.mutable_gpu_data());
	
	Dtype loss;
  caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);
  
  top[0]->mutable_cpu_data()[0] = loss / Dtype(num*height*width);
}

template <typename Dtype>
void MaxLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  if (Caffe::second_pass() == false)
	{
		backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width,channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
		
		const Dtype loss_weight = top[0]->cpu_diff()[0] / Dtype(num*height*width);
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_gpu_diff());
	}
}

template <typename Dtype>
void MaxLossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(MaxLossLayer);
}  // namespace caffe
