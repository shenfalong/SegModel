
#include <vector>

#include "caffe/layers/func/pad_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void PadImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	pad_ = this->layer_param_.convolution_param().pad();
}

template <typename Dtype>
void PadImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
  
	top[0]->Reshape(num,channels,height+2*pad_,width+2*pad_);
}

INSTANTIATE_CLASS(PadImageLayer);
REGISTER_LAYER_CLASS(PadImage);
}  // namespace caffe
