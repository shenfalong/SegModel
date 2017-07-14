#include <vector>

#include "caffe/layers/same_max_pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	top[0]->Reshape(num,channels,height,width);
	mask.Reshape(num,1,height,width);
}

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void SameMaxPoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(SameMaxPoolingLayer);
#endif

INSTANTIATE_CLASS(SameMaxPoolingLayer);
REGISTER_LAYER_CLASS(SameMaxPooling);
}  // namespace caffe
		
