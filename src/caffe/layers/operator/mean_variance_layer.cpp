
#include <vector>

#include "caffe/layers/operator/mean_variance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MeanVarianceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void MeanVarianceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	top[0]->Reshape(num,2*channels,1,1);
}



INSTANTIATE_CLASS(MeanVarianceLayer);
REGISTER_LAYER_CLASS(MeanVariance);
}  // namespace caffe
