#ifndef CAFFE_PARALLEL_LAYER_HPP_
#define CAFFE_PARALLEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {



template <typename Dtype>
class ParallelLayer : public Layer<Dtype> 
{
 public:
  explicit ParallelLayer(const LayerParameter& param) : Layer<Dtype>(param)  {}
  virtual ~ParallelLayer();
  virtual inline const char* type() const { return "Parallel"; }
  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,  const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
	virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
//-------------------------------------------  
	vector<shared_ptr<Layer<Dtype> > > unary_layer_;
	vector< vector<Blob<Dtype>* > > unary_bottom_vec_;
	vector< vector<Blob<Dtype>* > > unary_top_vec_;
//-------------------------------------------
	

	
};		


}  // namespace caffe

#endif 
