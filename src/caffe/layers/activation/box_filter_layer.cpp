#include <vector>

#include "caffe/layers/activation/box_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BoxFilterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	radius =this->layer_param_.crf_param().radius();
}

template <typename Dtype>
void BoxFilterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	buffer_.ReshapeLike(*bottom[0]);
	top[0]->ReshapeLike(*bottom[0]);
}

INSTANTIATE_CLASS(BoxFilterLayer);
REGISTER_LAYER_CLASS(BoxFilter);
}  // namespace caffe
		
