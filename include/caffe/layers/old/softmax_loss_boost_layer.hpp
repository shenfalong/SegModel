#ifndef CAFFE_SOFTMAX_WITH_LOSS_BOOST_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_LOSS_BOOST_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
class SoftmaxWithLossBoostLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxWithLossBoostLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "SoftmaxWithLossBoost"; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  
  
 protected:

  Blob<Dtype> prob_;
  
  
  shared_ptr<Layer<Dtype> > softmax_layer_;
 
  
  
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;

  bool has_ignore_label_;
 
  int ignore_label_;
 
  Blob<Dtype> counts_;
  Blob<Dtype> loss_;
  Blob<Dtype> flag;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_BOOST_LAYER_HPP_
