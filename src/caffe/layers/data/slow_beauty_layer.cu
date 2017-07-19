#include "caffe/solver.hpp"
#include <vector>
#include "caffe/util/format.hpp"

#include "caffe/layers/data/slow_style_data_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/data/slow_beauty_layer.hpp"

namespace caffe {

template <typename Dtype>
void SlowBeautyLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}

template <typename Dtype>
void SlowBeautyLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{

#if 1
	const Dtype correction = std::sqrt(Dtype(1) - std::pow(Dtype(0.999), Solver<Dtype>::iter() + 1)) / (Dtype(1.) - std::pow(Dtype(0.9), Solver<Dtype>::iter() + 1));
	
	adam_update_gpu(top[0]->count(), top[0]->mutable_gpu_diff(), history_0_.mutable_gpu_data(),history_1_.mutable_gpu_data(), 
	Dtype(0.9), Dtype(0.999), Dtype(1e-8), correction);     
#else
	caffe_gpu_add(top[0]->count(),Dtype(1),top[0]->gpu_diff(),Dtype(0.9),history_0_.gpu_data(),top[0]->mutable_gpu_diff());
	
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),history_0_.mutable_gpu_data());
#endif	
	caffe_gpu_add(top[0]->count(),Dtype(1),top[0]->gpu_data(),Dtype(-0.1),top[0]->gpu_diff(),top[0]->mutable_gpu_data());
	for (int i=0;i<top[0]->count();i++)
		top[0]->mutable_cpu_data()[i] = min(max(top[0]->cpu_data()[i],Dtype(-1)),Dtype(1));
}

template <typename Dtype>
void SlowBeautyLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(SlowBeautyLayer);
}  // namespace caffe
