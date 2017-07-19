#include <cfloat>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/context_pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"

using std::max;
using std::min;
using std::floor;
using std::ceil;

namespace caffe 
{
template <typename Dtype>
void ContextPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  ContextPoolingParameter context_pool_param = this->layer_param_.context_pooling_param();
  context_height_ = context_pool_param.context_h();
  context_width_ = context_pool_param.context_w();
}

template <typename Dtype>
void ContextPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
                                         const vector<Blob<Dtype>*>& top)
{
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_, height_,width_);
  max_idx_.Reshape(num_, channels_, height_,width_);
  arg_count_.Reshape(num_, channels_, height_,width_);
}

template <typename Dtype>
void ContextPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}


template <typename Dtype>
void ContextPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,  const vector<Blob<Dtype>*>& bottom)
{
}


#ifdef CPU_ONLY
STUB_GPU(ContextPoolingLayer);
#endif

INSTANTIATE_CLASS(ContextPoolingLayer);
REGISTER_LAYER_CLASS(ContextPooling);

}  // namespace caffe
