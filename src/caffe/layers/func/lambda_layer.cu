
#include <vector>

#include "caffe/layers/func/lambda_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void LambdaLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	all_layer_[layer_index_]->Forward_gpu(unary_bottom_vec_[layer_index_], top);
}

template <typename Dtype>
void LambdaLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	all_layer_[layer_index_]->Backward_gpu(top, unary_bottom_vec_[layer_index_]);
}

template <typename Dtype>
void LambdaLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	all_layer_[layer_index_]->SecForward_gpu(unary_bottom_vec_[layer_index_], top);
}

INSTANTIATE_LAYER_GPU_FUNCS(LambdaLayer);
}  // namespace caffe
