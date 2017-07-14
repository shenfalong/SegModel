
#include <vector>

#include "caffe/layers/func/parallel_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParallelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Forward_gpu(unary_bottom_vec_[i], unary_top_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}

template <typename Dtype>
void ParallelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Backward_gpu(unary_top_vec_[i],unary_bottom_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}
template <typename Dtype>
void ParallelLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->SecForward_gpu(unary_bottom_vec_[i], unary_top_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}
INSTANTIATE_LAYER_GPU_FUNCS(ParallelLayer);

}  // namespace caffe

