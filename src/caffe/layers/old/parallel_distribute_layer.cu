#include <vector>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/parallel_distribute_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParallelDistributeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype * bottom_data=bottom[0]->gpu_data();
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	caffe_copy(top[0]->count(),bottom_data,top[0]->mutable_gpu_data());
	
	bottom_data += top[0]->count();
	
 	for(int i=1;i<top.size();i++)
 	{
    CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
 		top[i]->mutable_gpu_data();
 		

    CUDA_CHECK(cudaMemcpyPeer(top[i]->mutable_gpu_data(),Caffe::GPUs[i],
                                   bottom_data,Caffe::GPUs[0],
																	 top[i]->count()*sizeof(Dtype)
																	 ));
	
		
																			 
 		bottom_data += top[i]->count();
 	}
 	CUDA_SYNCHRONIZE;
}
template <typename Dtype>
void ParallelDistributeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
	CUDA_SYNCHRONIZE;
}
INSTANTIATE_LAYER_GPU_FUNCS(ParallelDistributeLayer);
}  // namespace caffe
