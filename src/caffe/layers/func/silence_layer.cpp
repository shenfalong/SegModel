#include <vector>

#include "caffe/layers/func/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SilenceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void SilenceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_CLASS(SilenceLayer);
REGISTER_LAYER_CLASS(Silence);
}  // namespace caffe
		
