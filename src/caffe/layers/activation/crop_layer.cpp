#include <vector>

#include "caffe/layers/activation/crop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CropLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	pad_ = this->layer_param_.convolution_param().pad();
}

template <typename Dtype>
void CropLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
	top[0]->Reshape(num,channels,height-pad_*2,width-pad_*2);
}

INSTANTIATE_CLASS(CropLayer);
REGISTER_LAYER_CLASS(Crop);
}  // namespace caffe
		
