
#ifndef CAFFE_Be_GdLoss_LAYER_HPP_
#define CAFFE_Be_GdLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class BeGdLossLayer : public Layer<Dtype> {
 public:
  explicit BeGdLossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "BeGdLoss"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);



  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	Blob<Dtype> loss_d_;
 	Blob<Dtype> loss_g_;
 	Dtype sum_d;
 	Dtype sum_g;
 	
};

}  // namespace caffe

#endif  // CAFFE_BeGdLossLAYER_HPP_
