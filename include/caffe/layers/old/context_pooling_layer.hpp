#ifndef CAFFE_CONTEXT_POOLING_LAYER_HPP_
#define CAFFE_CONTEXT_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <algorithm>
#include <cfloat>
#include <vector>

namespace caffe {

template <typename Dtype>
class ContextPoolingLayer : public Layer<Dtype> {
 public:
  explicit ContextPoolingLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "ContextPooling"; }
  
  
  
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  
 
 protected:



	int num_;
  int channels_;
  int height_;
  int width_;
  int context_height_;
  int context_width_;
  Blob<int> max_idx_;
	Blob<int> arg_count_;
};


}  // namespace caffe

#endif  // CAFFE_CONTEXT_POOLING_LAYER_HPP_
