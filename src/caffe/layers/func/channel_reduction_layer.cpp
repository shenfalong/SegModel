
#include <vector>

#include "caffe/layers/func/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ChannelReductionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void ChannelReductionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	CHECK_EQ(channels%2,0);			
	top[0]->Reshape(num,channels/2,height,width);
}

INSTANTIATE_CLASS(ChannelReductionLayer);
REGISTER_LAYER_CLASS(ChannelReduction);
}  // namespace caffe
