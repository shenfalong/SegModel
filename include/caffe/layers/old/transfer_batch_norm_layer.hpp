
#ifndef CAFFE_TransferBatchNorm_LAYER_HPP_
#define CAFFE_TransferBatchNorm_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class TransferBatchNormLayer : public Layer<Dtype> {
 public:
  explicit TransferBatchNormLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual ~TransferBatchNormLayer();

  virtual inline const char* type() const { return "TransferBatchNorm"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:
 	Blob<Dtype> * mean_buffer_;
  Blob<Dtype> * var_buffer_;
  
  Blob<Dtype> * temp_mean_buffer_;
  Blob<Dtype> * temp_var_buffer_;
};

}  // namespace caffe

#endif  // CAFFE_TransferBatchNormLAYER_HPP_
