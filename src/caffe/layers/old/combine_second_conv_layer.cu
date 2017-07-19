
#include <vector>

#include "caffe/layers/combine_second_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
void CombineSecConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	conv1_layer_->Forward_gpu(conv1_bottom_vec_, conv1_top_vec_);//bottom[0] --> first_buffer_	
	//caffe_copy(first_buffer_->count(),bottom[0]->gpu_data(),first_buffer_->mutable_gpu_data());
	
	conv2_layer_->Forward_gpu(conv2_bottom_vec_, conv2_top_vec_);//bottom[0] --> sec_buffer_
	sec_layer_->Forward_gpu(sec_bottom_vec_, sec_top_vec_);//sec_buffer_, bottom[0] --> second_buffer_
	concat_layer_->Forward_gpu(concat_bottom_vec_, concat_top_vec_);//first_buffer_, second_buffer_ --> top[0]
}

template <typename Dtype>
void CombineSecConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	conv1_layer_->Forward_gpu(conv1_bottom_vec_, conv1_top_vec_);//bottom[0] --> first_buffer_	
	//caffe_copy(first_buffer_->count(),bottom[0]->gpu_data(),first_buffer_->mutable_gpu_data());	
	conv2_layer_->Forward_gpu(conv2_bottom_vec_, conv2_top_vec_);//bottom[0] --> sec_buffer_
	
	

	concat_layer_->Backward_gpu(concat_top_vec_, concat_bottom_vec_);//top[0] --> first_buffer_, second_buffer_

	caffe_gpu_set(bottom_buffer_->count(),Dtype(0),bottom_buffer_->mutable_gpu_diff());
	
	sec_layer_->Backward_gpu(sec_top_vec_, sec_bottom_vec_);//second_buffer_ --> sec_buffer_, first_buffer_	
	caffe_gpu_add(bottom_buffer_->count(),Dtype(1),bottom_buffer_->gpu_diff(),Dtype(1),bottom[0]->gpu_diff(),bottom_buffer_->mutable_gpu_diff());
	
	conv2_layer_->Backward_gpu(conv2_top_vec_, conv2_bottom_vec_);//sec_buffer_ --> bottom[0]
	caffe_gpu_add(bottom_buffer_->count(),Dtype(1),bottom_buffer_->gpu_diff(),Dtype(1),bottom[0]->gpu_diff(),bottom_buffer_->mutable_gpu_diff());
	
	conv1_layer_->Backward_gpu(conv1_top_vec_, conv1_bottom_vec_);//first_buffer_ --> bottom[0]
	//caffe_copy(bottom[0]->count(),first_buffer_->gpu_diff(),bottom[0]->mutable_gpu_diff());
	caffe_gpu_add(bottom_buffer_->count(),Dtype(1),bottom_buffer_->gpu_diff(),Dtype(1),bottom[0]->gpu_diff(),bottom_buffer_->mutable_gpu_diff());
	
	caffe_copy(bottom[0]->count(),bottom_buffer_->gpu_diff(),bottom[0]->mutable_gpu_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(CombineSecConvLayer);
}  // namespace caffe
