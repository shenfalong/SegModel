
#ifndef CAFFE_BatchScale_LAYER_HPP_
#define CAFFE_BatchScale_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class BatchScaleLayer : public Layer<Dtype> {
 public:
  explicit BatchScaleLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "BatchScale"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:

	Blob<Dtype> sum_;
 	
};

}  // namespace caffe

#endif  // CAFFE_BatchScaleLAYER_HPP_
