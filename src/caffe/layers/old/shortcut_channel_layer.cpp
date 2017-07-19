#include <vector> 

#include "caffe/layers/shortcut_channel_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
Blob<Dtype> * ShortcutChannelLayer<Dtype>::lambda = NULL;

template <typename Dtype>
Blob<Dtype> * ShortcutChannelLayer<Dtype>::all_one = NULL;

template <typename Dtype>
Blob<Dtype> * ShortcutChannelLayer<Dtype>::temp_channels = NULL;

template <typename Dtype>
void ShortcutChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  dropout_ratio_ = this->layer_param_.shortcut_param().dropout_ratio();
  //if (this->layer_param_.shortcut_param().has_groups())
	//	groups = this->layer_param_.shortcut_param().groups();
	//else
	//	groups = bottom[0]->channels();

  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1, bottom[0]->channels(), 1, 1));
    Dtype scale = this->layer_param_.shortcut_param().scale();
#ifndef CPU_ONLY
    caffe_gpu_set(this->blobs_[0]->count(),scale,this->blobs_[0]->mutable_gpu_data());
#endif
		
	  if (this->layer_param_.param_size() <= 0)
	  {
	  	this->lr_mult().push_back(1);
	  	this->decay_mult().push_back(1);
	  }	
	  else
	  {
	  	this->lr_mult().push_back(this->layer_param_.param(0).lr_mult());
	  	this->decay_mult().push_back(this->layer_param_.param(0).decay_mult());
	  }
	}	
  
  
  
  if (lambda == NULL)
   lambda = new Blob<Dtype>();
  if (all_one == NULL)
  	all_one = new Blob<Dtype>();
  if (temp_channels == NULL)
  	temp_channels = new Blob<Dtype>();	
}	

template <typename Dtype>
void ShortcutChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  if (bottom[0]->height() != bottom[1]->height() )
    LOG(FATAL)<<"wrong size";
  
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ShortcutChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void ShortcutChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

template <typename Dtype>
ShortcutChannelLayer<Dtype>::~ShortcutChannelLayer()
{
	if (lambda != NULL)
	{
		delete lambda;
		lambda = NULL;
	}
	if (all_one != NULL)
	{
		delete all_one;
		all_one = NULL;
	}
	if (temp_channels != NULL)
	{
		delete temp_channels;
		temp_channels = NULL;
	}
}

#ifdef CPU_ONLY
STUB_GPU(ShortcutChannelLayer);
#endif

INSTANTIATE_CLASS(ShortcutChannelLayer);
REGISTER_LAYER_CLASS(ShortcutChannel);
}  // namespace caffe
