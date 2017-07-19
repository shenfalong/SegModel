#ifndef CAFFE_GuidedCRF_LAYER_HPP_
#define CAFFE_GuidedCRF_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

// add shen
template <typename Dtype>
class GuidedCRFLayer : public Layer<Dtype> 
{
 public:
  explicit GuidedCRFLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
  virtual ~GuidedCRFLayer();
  virtual inline const char* type() const { return "GuidedCRF"; }
  
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  
 protected:
	
	void guided_filter_gpu(const int num,const int channels,const int maxStates,const int height,const int width,const Dtype *I,const Dtype * p,Dtype *output_p);
	
	int gpu_id_;
  
 	int radius;
  int maxIter;
  Dtype area;
  Dtype alpha;
  Dtype eps;
  Blob<Dtype> compatPot;  
  Blob<Dtype> filterPot;  
  Blob<Dtype> tempPot; 
  std::vector< Blob<Dtype> * > nodeBel; 

  Blob<Dtype> mean_I;
  Blob<Dtype> II;
  Blob<Dtype> mean_II;
  Blob<Dtype> var_I;
  Blob<Dtype> mean_p;
  Blob<Dtype> b;
  Blob<Dtype> mean_b;
  Blob<Dtype> inv_var_I;
  Blob<Dtype> buffer_image;
  Blob<Dtype> buffer_score;
  Blob<Dtype> buffer_image_image;
  Blob<Dtype> output_p1;
  Blob<Dtype> output_p2;

//---------------------------------------  
  Dtype * a;
  Dtype * mean_a;
  Dtype * cov_Ip;
  Dtype * Ip;
  Dtype * mean_Ip;
  Dtype * buffer_image_score;
//---------------------------------------

	vector<Blob<Dtype> *> myworkspace_;
};

}  // namespace caffe

#endif  // CAFFE_GuidedCRF_LAYER_HPP_
