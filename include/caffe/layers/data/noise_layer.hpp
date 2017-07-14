
#ifndef CAFFE_Noise_LAYER_HPP_
#define CAFFE_Noise_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class NoiseLayer : public Layer<Dtype> {
 public:
  explicit NoiseLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  

  virtual inline const char* type() const { return "Noise"; }
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	int num_;
 	int channels_;
 	int classes_;
 	Blob<Dtype>* top_buffer_;
 	
 	int gpu_id_;	
 	//int data_iter_;
 	
};

}  // namespace caffe

#endif  // CAFFE_NoiseLAYER_HPP_
