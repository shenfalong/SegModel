#ifndef CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {



template <typename Dtype>
class CuDNNBatchNormLayer : public Layer<Dtype> {
 public:
  explicit CuDNNBatchNormLayer(const LayerParameter& param) : Layer<Dtype>(param), handles_setup_(false)  {}
  virtual ~CuDNNBatchNormLayer();
	virtual inline const char* type() const { return "CuDNNBatchNorm"; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,   const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	
	
	double number_collect_sample;
	bool is_initialize;
		
  bool handles_setup_;
  static cudnnHandle_t *       handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t scale_bias_desc_;
  
  Blob<Dtype> weights;
  Blob<Dtype> bias;
  Blob<Dtype> * mean_buffer_;
  Blob<Dtype> * var_buffer_;
  
  
  int num_;
  int channels_;
  int height_;
  int width_;
  
  int gpu_id_;	
};


}  // namespace caffe

#endif 
