
#include <vector>

#include "caffe/layers/loss/gan_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/solver.hpp"
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
    		bottom_diff[ind] = prob_data[ind] -1;
			else
				bottom_diff[ind] = prob_data[ind];
    }
  }
}

template <typename Dtype>
void GANSoftmaxWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	softmax_layer_->Forward_gpu(softmax_bottom_vec_, softmax_top_vec_);

 
 	if (Caffe::gan_type() == "train_dnet")
 	{	
		int num = bottom[0]->num()/2;
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		for (int i=0;i<bottom[1]->count();i++)
		{
			CHECK_GE(bottom[1]->cpu_data()[i],0);
			CHECK_LE(bottom[1]->cpu_data()[i],channels-1);
		}
 	
 		
		forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data()+prob_.offset(num), bottom[1]->gpu_data(), loss_.mutable_gpu_data());
		
		Dtype loss;
		caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);

		top[0]->mutable_cpu_data()[0] = loss / loss_.count() * Dtype(1);
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		for (int i=0;i<bottom[1]->count();i++)
		{
			CHECK_GE(bottom[1]->cpu_data()[i],0);
			CHECK_LE(bottom[1]->cpu_data()[i],channels-1);
		}
 		
 	
		forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), loss_.mutable_gpu_data());
		
		Dtype loss;
		caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);

		top[0]->mutable_cpu_data()[0] = loss / loss_.count() * Dtype(0.1);
	}
	
	if (Solver<Dtype>::iter() % 100 == 0 && Caffe::gan_type() == "train_dnet")
			LOG(INFO)<<"softmax_loss = "<<top[0]->cpu_data()[0];
}

template <typename Dtype>
void GANSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	if (Caffe::gan_type() == "train_dnet")
 	{	
 		int num = bottom[0]->num()/2;
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		
		caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
		backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data()+prob_.offset(num), bottom[1]->gpu_data(), 
																					bottom[0]->mutable_gpu_diff()+bottom[0]->offset(num));


	
		const Dtype loss_weight =  top[0]->cpu_diff()[0] / loss_.count() * Dtype(1);
		caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
 	}
 	else
 	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
	
	

		backward_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width, num, channels, height*width, prob_.gpu_data(), bottom[1]->gpu_data(), bottom[0]->mutable_gpu_diff());


		const Dtype loss_weight = top[0]->cpu_diff()[0] / loss_.count() * Dtype(0.1);
		caffe_gpu_scal(prob_.count(), loss_weight, bottom[0]->mutable_gpu_diff());
	}
}

template <typename Dtype>
void GANSoftmaxWithLossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(GANSoftmaxWithLossLayer);
}  // namespace caffe
