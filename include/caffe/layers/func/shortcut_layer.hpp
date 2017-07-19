#ifndef CAFFE_SHORTCUT_LAYER_HPP_
#define CAFFE_SHORTCUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {

template <typename Dtype>
class ShortcutLayer : public Layer<Dtype> {
 public:
  explicit ShortcutLayer(const LayerParameter& param): Layer<Dtype>(param) {}
  virtual ~ShortcutLayer();
  virtual inline const char* type() const { return "Shortcut"; }

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,   const vector<Blob<Dtype>*>& bottom);
	virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
 protected:

  
};

}  // namespace caffe

#endif  
