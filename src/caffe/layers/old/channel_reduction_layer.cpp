
#include <vector>

#include "caffe/layers/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(ChannelReductionLayer);
#endif

INSTANTIATE_CLASS(ChannelReductionLayer);
REGISTER_LAYER_CLASS(ChannelReduction);
}  // namespace caffe
