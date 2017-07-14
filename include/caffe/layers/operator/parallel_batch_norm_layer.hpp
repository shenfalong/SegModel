#ifndef CAFFE_PARALLEL_BATCH_NORM_LAYER_HPP_
#define CAFFE_PARALLEL_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {



template <typename Dtype>
class ParallelBatchNormLayer : public Layer<Dtype> 
{
 public:
  explicit ParallelBatchNormLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual ~ParallelBatchNormLayer();
  virtual inline const char* type() const { return "ParallelBatchNorm"; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
	virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
	vector<Blob<Dtype> *> parallel_mean_buffer_;
	vector<Blob<Dtype> *> parallel_var_buffer_;  
};		

}  // namespace caffe

#endif 
