#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {



template <typename Dtype>
class BatchNormLayer : public Layer<Dtype> {
 public:
  explicit BatchNormLayer(const LayerParameter& param) : Layer<Dtype>(param), handles_setup_(false)  {}
  virtual ~BatchNormLayer();
	virtual inline const char* type() const { return "BatchNorm"; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);

  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,   const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:
 	
	
	double number_collect_sample;
	bool is_initialize;
		
  bool handles_setup_;


  Blob<Dtype> * mean_buffer_;
  Blob<Dtype> * var_buffer_;
 
  int gpu_id_;	
};


}  // namespace caffe

