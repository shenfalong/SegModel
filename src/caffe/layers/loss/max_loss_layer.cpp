
#include <vector>

#include "caffe/layers/loss/max_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void MaxLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	CHECK_EQ(bottom.size(),2);
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	loss_.Reshape(num,1,height,width);
	
	top[0]->Reshape(1,1,1,1);
}

INSTANTIATE_CLASS(MaxLossLayer);
REGISTER_LAYER_CLASS(MaxLoss);
}  // namespace caffe
