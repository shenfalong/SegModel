#ifndef CAFFE_ACCURACY_LAYER_HPP_
#define CAFFE_ACCURACY_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe 
{

template <typename Dtype>
class AccuracyLayer : public Layer<Dtype> 
{
 public:
  explicit AccuracyLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "Accuracy"; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:



  int top_k_;
  bool has_ignore_label_;
  int ignore_label_;

};

}  // namespace caffe

#endif  // CAFFE_ACCURACY_LAYER_HPP_
