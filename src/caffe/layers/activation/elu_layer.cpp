#include <algorithm>
#include <vector>

#include "caffe/layers/activation/elu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ELULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}


template <typename Dtype>
void ELULayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(ELULayer);
REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe

