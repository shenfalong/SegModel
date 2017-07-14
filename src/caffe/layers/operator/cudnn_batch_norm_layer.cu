#include <vector>
#include "caffe/layers/operator/cudnn_batch_norm_layer.hpp"
#define BN_EPS Dtype(1e-5)


namespace caffe {
template <typename Dtype>
static __global__ void linear_batch_norm_forward(int num,int channels,int height,int width,
													const Dtype *weight,const Dtype * in, const Dtype * bias, Dtype *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind] + bias[c];
  }
}

template <typename Dtype>
static __global__ void linear_batch_norm_backward(int num,int channels,int height,int width,
													const Dtype *weight,const Dtype * in, const Dtype * bias, Dtype *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind];
  }
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (Caffe::bn_state() == "frozen")
	{
		const int K = bottom[0]->channels();
		weights.Reshape(1,K,1,1);
		bias.Reshape(1,K,1,1);
	
		for(int c=0;c<K;c++)
		{
			weights.mutable_cpu_data()[c] = this->blobs_[0]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c]+ Dtype(CUDNN_BN_MIN_EPSILON)));
			bias.mutable_cpu_data()[c] = -this->blobs_[0]->cpu_data()[c]*this->blobs_[2]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c] + Dtype(CUDNN_BN_MIN_EPSILON)))
																								+this->blobs_[1]->cpu_data()[c];															
		}				
	} 	

	if (Caffe::number_collect_sample == 0 && Caffe::bn_state() == "learned")
	{
		caffe_gpu_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_gpu_data());
		caffe_gpu_set(this->blobs_[3]->count(),Dtype(0),this->blobs_[3]->mutable_gpu_data());
	}

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
   
	if (Caffe::bn_state() == "learned")
	{	
		double factor;
		if (Caffe::number_collect_sample == -1)
			factor = 0.01;
		else 
			factor = double(1)/double(Caffe::number_collect_sample+1);
	

		CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      factor,
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(CUDNN_BN_MIN_EPSILON),
		      mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	     
  }  
	else
	{

		linear_batch_norm_forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),bottom[0]->gpu_data(),bias.gpu_data(),top[0]->mutable_gpu_data()); 
/*
		CUDNN_CHECK(cudnnBatchNormalizationForwardInference(Caffe::parallel_cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(0.001)
		      ));	       	       
*/	         
	}	   	           
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	if (Caffe::bn_state() == "learned")
  {
		const Dtype* top_data = top[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (Caffe::frozen_param() == false)
		{		
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   			
		}
		else
		{		
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::zero,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),//not use
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   	
		}
  }    
  else
  {
  	linear_batch_norm_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),top[0]->gpu_diff(),bias.gpu_data(),bottom[0]->mutable_gpu_diff());  
  } 
}
template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom[0]->gpu_data(),
						bottom_desc_,bottom[0]->gpu_sec_diff(),
						top_desc_, top[0]->mutable_gpu_sec_diff(),
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_sec_diff(),//blobs_[1]->diff shoud be fixed
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   
}
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe
