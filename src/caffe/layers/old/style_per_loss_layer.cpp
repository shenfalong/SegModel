
#include <vector>

#include "caffe/layers/style_per_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StylePerLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	buffer_0_ = new Blob<Dtype>();
	buffer_1_ = new Blob<Dtype>();
	buffer_delta_ = new Blob<Dtype>();
	buffer_square_ = new Blob<Dtype>();
}

template <typename Dtype>
void StylePerLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 
	buffer_1_->Reshape(num,1,channels,channels);
	buffer_square_->Reshape(num,channels,height,width);
	top[0]->Reshape(1,1,1,1);
}

template <typename Dtype>
void StylePerLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void StylePerLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(StylePerLossLayer);
#endif

INSTANTIATE_CLASS(StylePerLossLayer);
REGISTER_LAYER_CLASS(StylePerLoss);
}  // namespace caffe
