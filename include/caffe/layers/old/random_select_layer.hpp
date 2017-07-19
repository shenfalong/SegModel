
#ifndef CAFFE_RandomSelect_LAYER_HPP_
#define CAFFE_RandomSelect_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class RandomSelectLayer : public Layer<Dtype> {
 public:
  explicit RandomSelectLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  ~RandomSelectLayer();

  virtual inline const char* type() const { return "RandomSelect"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);


  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
 protected:
	Blob<Dtype> * random_feature;
	Blob<Dtype> * bottom_buffer_;
	Blob<Dtype> * sec_buffer_;
	Blob<Dtype> * top_buffer_;
	
	
	shared_ptr<Layer<Dtype> > conv3x3_layer_0_;
	shared_ptr<Layer<Dtype> > conv3x3_layer_1_;
	
	vector<Blob<Dtype>*> conv3x3_bottom_vec_0_;
  vector<Blob<Dtype>*> conv3x3_top_vec_0_;
  
  vector<Blob<Dtype>*> conv3x3_bottom_vec_1_;
  vector<Blob<Dtype>*> conv3x3_top_vec_1_;
  
  int stride_;
};

}  // namespace caffe

#endif  // CAFFE_RandomSelectLAYER_HPP_
