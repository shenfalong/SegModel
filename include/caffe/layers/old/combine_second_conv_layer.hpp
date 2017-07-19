
#ifndef CAFFE_CombineSecConv_LAYER_HPP_
#define CAFFE_CombineSecConv_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class CombineSecConvLayer : public Layer<Dtype> {
 public:
  explicit CombineSecConvLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "CombineSecConv"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:
 	shared_ptr<Layer<Dtype> > conv1_layer_;
 	shared_ptr<Layer<Dtype> > conv2_layer_;
 	shared_ptr<Layer<Dtype> > sec_layer_;
 	shared_ptr<Layer<Dtype> > concat_layer_;
  
  vector<Blob<Dtype>*> conv1_bottom_vec_;
  vector<Blob<Dtype>*> conv1_top_vec_;
  
  vector<Blob<Dtype>*> conv2_bottom_vec_;
  vector<Blob<Dtype>*> conv2_top_vec_;
  
 	vector<Blob<Dtype>*> sec_bottom_vec_;
  vector<Blob<Dtype>*> sec_top_vec_;
  
  vector<Blob<Dtype>*> concat_bottom_vec_;
  vector<Blob<Dtype>*> concat_top_vec_;
//------------------------------------------	
	Blob<Dtype> * sec_buffer_;
	Blob<Dtype> * first_buffer_;
	Blob<Dtype> * second_buffer_;
	Blob<Dtype> * bottom_buffer_;
	int gpu_id_;	
//------------------------------------------  
  
};

}  // namespace caffe

#endif  // CAFFE_CombineSecConvLAYER_HPP_
