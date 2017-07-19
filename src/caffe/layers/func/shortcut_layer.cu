#include <vector>

#include "caffe/layers/func/shortcut_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ShortcutLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	caffe_gpu_add(bottom[0]->count(),Dtype(1),bottom[0]->gpu_data(), Dtype(1),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());				
}

template <typename Dtype>
void ShortcutLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();


	caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff()); 
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),bottom[1]->mutable_gpu_diff());
}
template <typename Dtype>
void ShortcutLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	caffe_gpu_add(bottom[0]->count(),Dtype(1),bottom[0]->gpu_sec_diff(), Dtype(1),bottom[1]->gpu_sec_diff(),
						top[0]->mutable_gpu_sec_diff());		
}
INSTANTIATE_LAYER_GPU_FUNCS(ShortcutLayer);
}  // namespace caffe
