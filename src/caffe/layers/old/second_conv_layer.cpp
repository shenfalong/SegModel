
#include <vector>

#include "caffe/layers/second_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SecConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void SecConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels_sec = bottom[0]->channels();
  int channels_x = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  CHECK_EQ(channels_sec%channels_x, 0);
  
  top[0]->Reshape(num, channels_sec/channels_x, height, width);
}

template <typename Dtype>
void SecConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void SecConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(SecConvLayer);
#endif

INSTANTIATE_CLASS(SecConvLayer);
REGISTER_LAYER_CLASS(SecConv);
}  // namespace caffe
