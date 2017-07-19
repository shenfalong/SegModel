
#ifndef CAFFE_Gate_LAYER_HPP_
#define CAFFE_Gate_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class GateLayer : public Layer<Dtype> {
 public:
  explicit GateLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "Gate"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:
 	cudnnTensorDescriptor_t bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  cudnnConvolutionDescriptor_t conv_descs_;
  
  Blob<Dtype> * buffer_top_;
  Blob<Dtype> weight_;
  int gpu_id_;
  
  
 	
};

}  // namespace caffe

#endif  // CAFFE_GateLAYER_HPP_
