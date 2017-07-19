
#include <vector>

#include "caffe/layers/loss/gradient_penalty_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void GradientPenaltyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	CHECK_EQ(bottom.size(),1);
	//CHECK_EQ(channels,1);
	
	caffe_gpu_sum(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());	
}

template <typename Dtype>
void GradientPenaltyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	if (Caffe::second_pass() == false)
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		Dtype loss_weights_ = top[0]->cpu_diff()[0];
		caffe_gpu_set(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());	
	}
	else
	{
		caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
	}
}
template <typename Dtype>
void GradientPenaltyLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(GradientPenaltyLayer);
}  // namespace caffe
