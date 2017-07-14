
#include <vector>

#include "caffe/layers/transfer_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	mean_buffer_ = new Blob<Dtype>();
	var_buffer_ = new Blob<Dtype>();
}

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  mean_buffer_->Reshape(num,channels,1,1);
  var_buffer_->Reshape(num,channels,1,1);
  
	top[0]->Reshape(num/2,channels,height,width);
}

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void TransferBatchNormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

template <typename Dtype>
TransferBatchNormLayer<Dtype>::~TransferBatchNormLayer() 
{
	delete mean_buffer_;
	delete var_buffer_;
}

#ifdef CPU_ONLY
STUB_GPU(TransferBatchNormLayer);
#endif

INSTANTIATE_CLASS(TransferBatchNormLayer);
REGISTER_LAYER_CLASS(TransferBatchNorm);
}  // namespace caffe
