#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ = this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) 
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	//CHECK_EQ(bottom.size(),2);
	CHECK_EQ(bottom[1]->channels(),1);
	CHECK_EQ(bottom[0]->num(),bottom[1]->num());
	CHECK_EQ(bottom[0]->height(),bottom[1]->height());
	CHECK_EQ(bottom[0]->width(),bottom[1]->width());
	
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  top[0]->Reshape(1,1,1,1);
  loss_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
  counts_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}


INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
