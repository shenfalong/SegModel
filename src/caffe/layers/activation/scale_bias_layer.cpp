
#include <vector>

#include "caffe/layers/activation/scale_bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ScaleBiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

	classes_ = this->layer_param_.noise_param().classes();
	
	if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {

    this->blobs_.resize(2);
	
  	int channels = bottom[0]->channels();  
    this->blobs_[0].reset(new Blob<Dtype>(classes_,channels,1,1));
   	this->blobs_[1].reset(new Blob<Dtype>(classes_,channels,1,1));
		caffe_set(this->blobs_[0]->count(),Dtype(1),this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_cpu_data());
		
    if (this->lr_mult().size() == 0)
    {
    	this->lr_mult().push_back(1);
    	this->decay_mult().push_back(1);
    }	
  }
}

template <typename Dtype>
void ScaleBiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}


INSTANTIATE_CLASS(ScaleBiasLayer);
REGISTER_LAYER_CLASS(ScaleBias);
}  // namespace caffe
