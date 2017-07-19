#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>


#include "caffe/layers/activation/image_resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ImageResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	interp_ratio = this->layer_param().interp_param().interp_ratio();
}

template <typename Dtype>
void ImageResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
  int width = bottom[0]->width();
 
	top[0]->Reshape(num,channels,height/interp_ratio,width/interp_ratio);
}



INSTANTIATE_CLASS(ImageResizeLayer);
REGISTER_LAYER_CLASS(ImageResize);
}  // namespace caffe
		
