
#ifndef CAFFE_InterpData_LAYER_HPP_
#define CAFFE_InterpData_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class InterpDataLayer : public Layer<Dtype> {
 public:
  explicit InterpDataLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "InterpData"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual inline void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual inline void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	Blob<Dtype> coef_;
};

}  // namespace caffe

#endif  // CAFFE_InterpDataLAYER_HPP_
