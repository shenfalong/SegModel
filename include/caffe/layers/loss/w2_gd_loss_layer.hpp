
#ifndef CAFFE_W2GdLoss_LAYER_HPP_
#define CAFFE_W2GdLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class W2GdLossLayer : public Layer<Dtype> {
 public:
  explicit W2GdLossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "W2GdLoss"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	Blob<Dtype> loss_;
 
 	Blob<Dtype> loss_g_;
 	Blob<Dtype> loss_d_;
 	Blob<Dtype> loss_c_;
 	Blob<Dtype> prob_;
 	Blob<Dtype> mask_;
 	Blob<Dtype> count_;
 	
};

}  // namespace caffe

#endif  // CAFFE_W2GdLossLAYER_HPP_
