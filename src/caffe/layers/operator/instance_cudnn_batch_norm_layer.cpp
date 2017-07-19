#include <vector>

#include "caffe/layers/operator/instance_cudnn_batch_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void InstanceCuDNNBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	

  cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
  cudnn::createTensor4dDesc<Dtype>(&top_desc_);
  cudnn::createTensor4dDesc<Dtype>(&scale_bias_desc_);
  
  
  if (this->blobs_.size() == 4)
  	LOG(INFO)<<"skip initialization";
  else 
  {
    const int K = bottom[0]->channels();
    this->blobs_.resize(4);
    for(int i=0;i<this->blobs_.size();i++)
    {
      this->blobs_[i].reset(new Blob<Dtype>());
      this->blobs_[i]->Reshape(1,K,1,1);
    }
    Dtype std = 0.02;
    //caffe_rng_gaussian<Dtype>(this->blobs_[0]->count(), Dtype(1), std, this->blobs_[0]->mutable_cpu_data());
    //caffe_rng_gaussian<Dtype>(this->blobs_[1]->count(), Dtype(0), std, this->blobs_[1]->mutable_cpu_data());
    caffe_set(this->blobs_[0]->count(),Dtype(1),this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_cpu_data());
    caffe_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_cpu_data());
    caffe_set(this->blobs_[3]->count(),Dtype(1),this->blobs_[3]->mutable_cpu_data());
		

		if (this->layer_param_.param_size() == 2)
	  { 
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
	  }	
		else if(this->layer_param_.param_size() == 0)
		{		
			this->lr_mult().push_back(1);
		  this->decay_mult().push_back(1);
		  this->lr_mult().push_back(1);
		  this->decay_mult().push_back(1);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		} 
		else 
			LOG(FATAL)<<"wrong lr_mult setting";
  }

  
  is_initialize = false;
}

template <typename Dtype>
void InstanceCuDNNBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

  const int N = bottom[0]->num();
  const int K = bottom[0]->channels();
  const int H = bottom[0]->height();
  const int W = bottom[0]->width();

	if (N == num_ && K == channels_ && H == height_ && W == width_)
		return;
	
	num_=N;
	channels_=K;
	height_=H;
	width_=W;


  top[0]->Reshape(N,K,H,W);   

  savedmean.Reshape(N,K,1,1);
  savedinvvariance.Reshape(N,K,1,1);

  cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, 1, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&top_desc_, 1, K, H, W);
  cudnn::setTensor4dDesc<Dtype>(&scale_bias_desc_,  1, K, 1, 1);

}

template <typename Dtype>
InstanceCuDNNBatchNormLayer<Dtype>::~InstanceCuDNNBatchNormLayer() 
{
	cudnnDestroyTensorDescriptor(this->bottom_desc_);
	cudnnDestroyTensorDescriptor(this->top_desc_);
	cudnnDestroyTensorDescriptor(this->scale_bias_desc_);
}

INSTANTIATE_CLASS(InstanceCuDNNBatchNormLayer);
REGISTER_LAYER_CLASS(InstanceCuDNNBatchNorm);
}  // namespace caffe
