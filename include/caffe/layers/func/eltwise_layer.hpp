#ifndef CAFFE_ELTWISE_LAYER_HPP_
#define CAFFE_ELTWISE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Compute elementwise operations, such as product and sum,
 *        along multiple input Blobs.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */
template <typename Dtype>
class EltwiseLayer : public Layer<Dtype> 
{
 public:
  explicit EltwiseLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "Eltwise"; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top  , const vector<Blob<Dtype>*>& bottom);
	virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

 protected:


  string op_;
  vector<Dtype> coeffs_;
  vector<bool> backwards_;
  
  Blob<int> max_idx_;

  bool stable_prod_grad_;
};

}  // namespace caffe

#endif  // CAFFE_ELTWISE_LAYER_HPP_
