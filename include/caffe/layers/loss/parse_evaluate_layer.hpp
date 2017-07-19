#ifndef CAFFE_PARSE_EVALUATE_LAYER_HPP_
#define CAFFE_PARSE_EVALUATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class ParseEvaluateLayer : public Layer<Dtype> {
 public:
  explicit ParseEvaluateLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "ParseEvaluate"; }
  
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:

  // number of total labels
  int num_labels_;
  // store ignored labels
  std::set<Dtype> ignore_labels_;
};

}

#endif  // CAFFE_LOSS_LAYER_HPP_
