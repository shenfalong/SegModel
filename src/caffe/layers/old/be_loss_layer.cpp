
#include <vector>

#include "caffe/layers/be_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BeLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//k_ = 0.5;
	
}

template <typename Dtype>
void BeLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
#if 0
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  CHECK_EQ(num%2,0);
  
  real_loss_.Reshape(num/2,channels,height,width);
  fake_loss_.Reshape(num/2,channels,height,width);
  
	top[0]->Reshape(1,1,1,1);
#endif	
}

template <typename Dtype>
void BeLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void BeLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(BeLossLayer);
#endif

INSTANTIATE_CLASS(BeLossLayer);
REGISTER_LAYER_CLASS(BeLoss);
}  // namespace caffe
