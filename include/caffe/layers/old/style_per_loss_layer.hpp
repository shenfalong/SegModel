
#ifndef CAFFE_StylePerLoss_LAYER_HPP_
#define CAFFE_StylePerLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class StylePerLossLayer : public Layer<Dtype> {
 public:
  explicit StylePerLossLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "StylePerLoss"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:
 	Blob<Dtype> * buffer_0_;
 	Blob<Dtype> * buffer_1_;
 	Blob<Dtype> * buffer_delta_;
 	Blob<Dtype> * buffer_square_;
};

}  // namespace caffe

#endif  // CAFFE_StylePerLossLAYER_HPP_
