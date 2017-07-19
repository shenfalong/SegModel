
#include <vector>

#include "caffe/layers/func/rgb_gray_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RGBGRAYLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void RGBGRAYLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	CHECK_EQ(channels,3);  
  
	top[0]->Reshape(num,1,height,width);
}

INSTANTIATE_CLASS(RGBGRAYLayer);
REGISTER_LAYER_CLASS(RGBGRAY);
}  // namespace caffe
