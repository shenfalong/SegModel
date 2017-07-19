
#include <vector>

#include "caffe/layers/activation/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BiasLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {

    this->blobs_.resize(1);
	
  	int channels = bottom[0]->channels();  
    this->blobs_[0].reset(new Blob<Dtype>(1,channels,1,1));
		caffe_set(this->blobs_[0]->count(),Dtype(0),this->blobs_[0]->mutable_cpu_data());
		
    if (this->lr_mult().size() == 0)
    {
    	this->lr_mult().push_back(1);
    	this->decay_mult().push_back(1);
    }	
  }
}

template <typename Dtype>
void BiasLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}


INSTANTIATE_CLASS(BiasLayer);
REGISTER_LAYER_CLASS(Bias);
}  // namespace caffe
