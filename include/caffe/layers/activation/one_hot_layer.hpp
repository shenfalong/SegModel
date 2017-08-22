
#ifndef CAFFE_OneHot_LAYER_HPP_
#define CAFFE_OneHot_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class OneHotLayer : public Layer<Dtype> {
 public:
  explicit OneHotLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "OneHot"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 
 	int classes_;
 	
};

}  // namespace caffe

#endif  // CAFFE_OneHotLAYER_HPP_
