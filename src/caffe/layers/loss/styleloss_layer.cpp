
#include <vector>

#include "caffe/layers/loss/styleloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StyleLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	buffer_0_ = new Blob<Dtype>();
	buffer_1_ = new Blob<Dtype>();
	buffer_delta_ = new Blob<Dtype>();
	buffer_square_ = new Blob<Dtype>();
}

template <typename Dtype>
void StyleLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	buffer_0_->Reshape(num,1,channels,channels);
	buffer_1_->Reshape(num,1,channels,channels);
	buffer_delta_->Reshape(num,1,channels,channels);
	buffer_square_->Reshape(num,1,channels,channels);
	top[0]->Reshape(1,1,1,1);
}


INSTANTIATE_CLASS(StyleLossLayer);
REGISTER_LAYER_CLASS(StyleLoss);
}  // namespace caffe
