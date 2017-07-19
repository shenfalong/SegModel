#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/activation/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (this->layer_param_.pooling_param().global_pool() == false)
	{
		kernel_size_ = this->layer_param_.pooling_param().kernel_size();	
		pad_ = this->layer_param_.pooling_param().pad();
		stride_ = this->layer_param_.pooling_param().stride();
	}	
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
  

	if (this->layer_param_.pooling_param().global_pool() == false)
	{
		pooled_height_ = (height + 2 * pad_ - kernel_size_) / stride_ + 1;
		pooled_width_ = (width + 2 * pad_ - kernel_size_) / stride_ + 1;
	}
	else
	{
		pooled_height_ = 1;
		pooled_width_ = 1;
	}

  
  top[0]->Reshape(num, channels, pooled_height_, pooled_width_);
    
  if (this->layer_param_.pooling_param().pool() == "max")
     max_idx_.Reshape(num, channels, pooled_height_, pooled_width_);
}



INSTANTIATE_CLASS(PoolingLayer);
REGISTER_LAYER_CLASS(Pooling);

}  // namespace caffe
