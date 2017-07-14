
#ifndef CAFFE_Lambda_LAYER_HPP_
#define CAFFE_Lambda_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class LambdaLayer : public Layer<Dtype> {
 public:
  explicit LambdaLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual inline const char* type() const { return "Lambda"; }
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	
 	int gpu_id_;
 	vector<shared_ptr<Layer<Dtype> > > all_layer_;
 	vector< vector<Blob<Dtype>* > > unary_bottom_vec_;
 	int layer_index_;
};

}  // namespace caffe

#endif  // CAFFE_LambdaLAYER_HPP_
