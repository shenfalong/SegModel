
#include <vector>

#include "caffe/layers/loss/euclideanloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	buffer_delta_ = new Blob<Dtype>();
	buffer_square_ = new Blob<Dtype>();
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	
	buffer_delta_->Reshape(num,channels,height,width);
	buffer_square_->Reshape(num,channels,height,width);
	top[0]->Reshape(1,1,1,1);
}


INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EuclideanLoss);
}  // namespace caffe
