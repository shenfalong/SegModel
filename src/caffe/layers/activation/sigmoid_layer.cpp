#include <cmath>
#include <vector>

#include "caffe/layers/activation/sigmoid_layer.hpp"

namespace caffe {


template <typename Dtype>
void SigmoidLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}


template <typename Dtype>
void SigmoidLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  top[0]->ReshapeLike(*bottom[0]);
}



INSTANTIATE_CLASS(SigmoidLayer);
REGISTER_LAYER_CLASS(Sigmoid);

}  // namespace caffe
