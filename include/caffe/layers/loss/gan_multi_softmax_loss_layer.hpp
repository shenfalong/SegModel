
#ifndef CAFFE_GANMultiSoftmaxWithLoss_LAYER_HPP_
#define CAFFE_GANMultiSoftmaxWithLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class GANMultiSoftmaxWithLossLayer : public Layer<Dtype> {
 public:
  explicit GANMultiSoftmaxWithLossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "GANMultiSoftmaxWithLoss"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	Blob<Dtype> loss_;
 	
 	Blob<Dtype> prob_;
  
  
  shared_ptr<Layer<Dtype> > softmax_layer_;
 
  
  
  vector<Blob<Dtype>*> softmax_bottom_vec_;
  vector<Blob<Dtype>*> softmax_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_GANMultiSoftmaxWithLossLAYER_HPP_
