#ifndef CAFFE_SoftmaxCrossEntropyLoss_LAYER_HPP_
#define CAFFE_SoftmaxCrossEntropyLoss_LAYER_HPP

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class SoftmaxCrossEntropyLossLayer : public Layer<Dtype> {
 public:
  explicit SoftmaxCrossEntropyLossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "SoftmaxCrossEntropyLoss"; }
	
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

  
  Blob<Dtype> loss_;
 	
};

}  // namespace caffe

#endif  // CAFFE_SoftmaxCrossEntropyLoss_LAYER_HPP_
