
#include <vector>

#include "caffe/layers/func/trivial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TrivialLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(bottom[i]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	}
}

template <typename Dtype>
void TrivialLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(top[i]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	}
}

template <typename Dtype>
void TrivialLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(bottom[i]->count(),bottom[0]->gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(TrivialLayer);
}  // namespace caffe
