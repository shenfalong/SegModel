#include <algorithm>
#include <vector>

#include "caffe/layers/activation/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int channels = bottom[0]->channels();
	
	
	sum_multiplier_.Reshape(1, channels, 1, 1);
	caffe_set(sum_multiplier_.count(), Dtype(1), sum_multiplier_.mutable_cpu_data());
}


template <typename Dtype>
void SoftmaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  top[0]->ReshapeLike(*bottom[0]);
  
	
  scale_.Reshape(num,1,height,width); 
}



INSTANTIATE_CLASS(SoftmaxLayer);
REGISTER_LAYER_CLASS(Softmax);

}  // namespace caffe
