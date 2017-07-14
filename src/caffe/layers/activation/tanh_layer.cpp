
#include <vector>

#include "caffe/layers/activation/tanh_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TanHLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void TanHLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}


INSTANTIATE_CLASS(TanHLayer);
REGISTER_LAYER_CLASS(TanH);
}  // namespace caffe
