#include <vector>

#include "caffe/layers/activation/imresize_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

namespace caffe {

template <typename Dtype>
void ImresizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	stride = this->layer_param().interp_param().stride();
	kernel_size = this->layer_param().interp_param().kernel_size();
	num_classes = this->layer_param().interp_param().num_classes();
}

template <typename Dtype>
void ImresizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  if (top.size() == 2)
	{
		top[0]->Reshape(num,num_classes,height/stride,width/stride);
		top[1]->Reshape(num,1,height/stride,width/stride);
	}
	else if (top.size() == 1)
		top[0]->Reshape(num,1,height/stride,width/stride);
}


INSTANTIATE_CLASS(ImresizeLayer);
REGISTER_LAYER_CLASS(Imresize);
}  // namespace caffe
		
