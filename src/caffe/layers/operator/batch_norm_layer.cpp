#include <vector>

#include "caffe/layers/operator/batch_norm_layer.hpp"

namespace caffe {

template <typename Dtype>
void BatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	mean_buffer_ = new Blob<Dtype>();
	var_buffer_ = new Blob<Dtype>();
	
  if (this->blobs_.size() == 2)
  	LOG(INFO)<<"skip initialization";
  else 
  {
    const int K = bottom[0]->channels();
    this->blobs_.resize(2);
    for(int i=0;i<this->blobs_.size();i++)
    {
      this->blobs_[i].reset(new Blob<Dtype>());
      this->blobs_[i]->Reshape(1,K,1,1);
    }
    caffe_set(this->blobs_[0]->count(),Dtype(1),this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_cpu_data());
		


		if (this->layer_param_.param_size() == 0)
		{		
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		} 
  }
}

template <typename Dtype>
void BatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	top[0]->ReshapeLike(*bottom[0]);

	
	mean_buffer_->Reshape(1,channels,1,1);
  var_buffer_->Reshape(1,channels,1,1);
  
}

template <typename Dtype>
BatchNormLayer<Dtype>::~BatchNormLayer() 
{
	delete mean_buffer_;
	delete var_buffer_;
}

INSTANTIATE_CLASS(BatchNormLayer);
REGISTER_LAYER_CLASS(BatchNorm);
}  // namespace caffe
