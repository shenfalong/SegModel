#ifndef CAFFE_PARALLEL_DISTRIBUTE_LAYER_HPP_
#define CAFFE_PARALLEL_DISTRIBUTE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ParallelDistributeLayer : public Layer<Dtype> {
 public:
  explicit ParallelDistributeLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "ParallelDistribute"; }
  
  
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,  const vector<Blob<Dtype>*>& top);
	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  


 protected:
};

}  // namespace caffe

#endif  
