
#ifndef CAFFE_FixedConv_LAYER_HPP_
#define CAFFE_FixedConv_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class FixedConvLayer : public Layer<Dtype> {
 public:
  explicit FixedConvLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "FixedConv"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:

//------------------------------------------	
	Blob<Dtype> * diff_weight_buffer_;
	Blob<Dtype> * all_one_;
	int gpu_id_;	
//------------------------------------------
	int kernel_size_;
	int filter_stride_;

};

}  // namespace caffe

#endif  // CAFFE_FixedConvLAYER_HPP_
