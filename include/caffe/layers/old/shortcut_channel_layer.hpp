#ifndef CAFFE_SHORTCUT_CHANNEL_LAYER_HPP_
#define CAFFE_SHORTCUT_CHANNEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ShortcutChannelLayer : public Layer<Dtype> {
 public:
  explicit ShortcutChannelLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual ~ShortcutChannelLayer();
  virtual inline const char* type() const { return "ShortcutChannel"; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,   const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,   const vector<Blob<Dtype>*>& bottom);
	
 protected:

      
  static Blob<Dtype> * lambda;
  static Blob<Dtype> * all_one;
  static Blob<Dtype> * temp_channels;
  
  Dtype multiplier;
  Dtype dropout_ratio_;
  int mask;
  int groups;
  
};

}  // namespace caffe

#endif  
