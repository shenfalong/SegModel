#include <algorithm>
#include <vector>

#include "caffe/layers/activation/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	negative_slope = this->layer_param_.relu_param().negative_slope();
}


template <typename Dtype>
void ReLULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	
	
  top[0]->ReshapeLike(*bottom[0]);
  flag.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}

INSTANTIATE_CLASS(ReLULayer);
REGISTER_LAYER_CLASS(ReLU);
}  // namespace caffe
