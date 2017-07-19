// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/activation/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DropoutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
}

template <typename Dtype>
void DropoutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();



	top[0]->ReshapeLike(*bottom[0]);
	
  rand_vec_.Reshape(1,channels,1,1);
}


INSTANTIATE_CLASS(DropoutLayer);
REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
