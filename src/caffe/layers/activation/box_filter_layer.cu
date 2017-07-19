#include <vector>

#include "caffe/layers/activation/box_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void BoxFilterLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
  box_filter_gpu(num,channels,height,width,radius,bottom[0]->gpu_data(),top[0]->mutable_gpu_data(),buffer_.mutable_gpu_data());

}

template <typename Dtype>
void BoxFilterLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	box_filter_gpu(num,channels,height,width,radius,top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff(),buffer_.mutable_gpu_data());
}

template <typename Dtype>
void BoxFilterLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}

INSTANTIATE_LAYER_GPU_FUNCS(BoxFilterLayer);
}  // namespace caffe
		
