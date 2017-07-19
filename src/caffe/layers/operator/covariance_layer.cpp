
#include <vector>

#include "caffe/layers/operator/covariance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CovarianceLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void CovarianceLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	top[0]->Reshape(num,channels*channels,1,1);
}

INSTANTIATE_CLASS(CovarianceLayer);
REGISTER_LAYER_CLASS(Covariance);
}  // namespace caffe
