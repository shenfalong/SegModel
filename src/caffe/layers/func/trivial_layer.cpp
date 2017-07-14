
#include <vector>

#include "caffe/layers/func/trivial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrivialLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void TrivialLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			top[i]->ReshapeLike(*bottom[i]);
	}
}

INSTANTIATE_CLASS(TrivialLayer);
REGISTER_LAYER_CLASS(Trivial);
}  // namespace caffe
