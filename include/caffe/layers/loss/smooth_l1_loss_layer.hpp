#ifndef CAFFE_SmoothL1Loss_LAYER_HPP_
#define CAFFE_SmoothL1Loss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class SmoothL1LossLayer : public Layer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "SmoothL1Loss"; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
 protected:


  Blob<Dtype> counts_;
  Blob<Dtype> loss_;
  
  int ignore_value;
};

}  // namespace caffe

#endif  // CAFFE_SmoothL1Loss_LAYER_HPP_
