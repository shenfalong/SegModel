#include <functional>
#include <utility>
#include <vector>

#include "caffe/layers/loss/accuracy_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe 
{

template <typename Dtype>
void AccuracyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  top_k_ = this->layer_param_.accuracy_param().top_k();

  has_ignore_label_ = this->layer_param_.accuracy_param().has_ignore_label();
  if (has_ignore_label_) 
    ignore_label_ = this->layer_param_.accuracy_param().ignore_label();
}

template <typename Dtype>
void AccuracyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  top[0]->Reshape(1,1,1,1);
}





INSTANTIATE_CLASS(AccuracyLayer);
REGISTER_LAYER_CLASS(Accuracy);

}  // namespace caffe
