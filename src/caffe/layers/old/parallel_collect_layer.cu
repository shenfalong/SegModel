#include <vector>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/parallel_collect_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParallelCollectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	Dtype * top_data=top[0]->mutable_gpu_data();
	caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top_data);
	top_data += bottom[0]->count();
 	for(int i=1;i<bottom.size();i++)
 	{			
	
    CUDA_CHECK(cudaMemcpyPeer(top_data,Caffe::GPUs[0],
                                  bottom[i]->gpu_data(),Caffe::GPUs[i],
																	bottom[i]->count()*sizeof(Dtype)
																	));

 		top_data += bottom[i]->count();
 		CUDA_CHECK(cudaDeviceSynchronize());
 	}
 	CUDA_SYNCHRONIZE;
}
template <typename Dtype>
void ParallelCollectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	const Dtype * top_diff=top[0]->gpu_diff();
	caffe_copy(bottom[0]->count(),top_diff,bottom[0]->mutable_gpu_diff());
	top_diff += bottom[0]->count();
 	for(int i=1;i<bottom.size();i++)
 	{
    CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
    CUDA_CHECK(cudaMemcpyPeer(bottom[i]->mutable_gpu_diff(),Caffe::GPUs[i],
                                  top_diff,Caffe::GPUs[0],
																	bottom[i]->count()*sizeof(Dtype)
																	));
 		top_diff += bottom[i]->count();
 	}		
 	CUDA_SYNCHRONIZE;
}

INSTANTIATE_LAYER_GPU_FUNCS(ParallelCollectLayer);
}  // namespace caffe
