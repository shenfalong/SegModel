
#include <vector>

#include "caffe/layers/loss/multi_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void forward_kernel(const int num_spatial, const int num, const int channels, const int spatial_dim,
          const Dtype* prob_data, const Dtype* label,  Dtype* loss) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
   		

    loss[index] = -log(prob_data[(n * channels + label_value) * spatial_dim + s]);
  }
}

template <typename Dtype>
static __global__ void backward_kernel(const int num_spatial,const int num, const int channels, const int spatial_dim, 
					const Dtype* prob_data, const Dtype* label,  Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num_spatial) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    int label_value = static_cast<int>(label[index]);
		
  	for (int c = 0; c < channels; ++c) 
    {
    	int ind = (n*channels+c)*spatial_dim+s;
    	if (c == label_value)
    		bottom_diff[ind] = prob_data[ind] - 1;
			else
				bottom_diff[ind] = prob_data[ind];
    }
  }
}


template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	softmax_layer_->Forward_gpu(softmax_bottom_vec_, softmax_top_vec_);
	
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	//LOG(INFO)<<"num = "<<num<<", channels = "<<channels;
	
	forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), loss_.mutable_gpu_data());
	
	Dtype loss;
	caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);

	top[0]->mutable_cpu_data()[0] = loss / loss_.count();
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
	backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), 
																				bottom[0]->mutable_gpu_diff());

	const Dtype loss_weight =  top[0]->cpu_diff()[0] / loss_.count();
	caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
}

template <typename Dtype>
void MultiSoftmaxWithLossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(MultiSoftmaxWithLossLayer);
}  // namespace caffe
