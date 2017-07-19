#include <vector>

#include "caffe/layers/activation/interp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	interp_ratio = this->layer_param_.interp_param().interp_ratio();
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (bottom.size() == 2)
  	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[1]->height(),bottom[1]->width());
  else
  	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),interp_ratio*bottom[0]->height(),interp_ratio*bottom[0]->width());
}


INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(Interp);
}  // namespace caffe
