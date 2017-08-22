// Copyright Yangqing Jia 2013

#include <set>
#include <string>
#include <vector>

#include "caffe/proto/caffe.pb.h"
#include "caffe/layer_factory.hpp"
#include "caffe/net.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/format.hpp"
#include "caffe/layers/func/parallel_layer.hpp"
#include <boost/thread.hpp>
using std::pair;
using std::map;
using std::set;

namespace caffe {


template <typename Dtype>
void Net<Dtype>::BcastData()
{	
	if (NGPUS > 1)
  {
		for (int i = 0; i < layers_.size(); i++)
		{	
			for (int j = 0;j < layers_[i]->blobs().size();j++)
			{			
				for (int k = 0; k < NGPUS; k++) 
				{
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[k]));
					ncclBcast((void *)layers_[i]->parallel_blobs()[j*NGPUS+k]->mutable_gpu_data(),layers_[i]->parallel_blobs()[j*NGPUS+k]->count(),
														ncclFloat,0,Caffe::comms(k),NULL);	
				}	
			}
		}	
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		//-----------------------------------------
	}
}
template <typename Dtype>
void Net<Dtype>::ReduceDiff()
{	
	if (NGPUS > 1)
  {	
		for (int i = layers_.size() - 1; i >= 0; i--)
		{						
			for (int j = 0;j < layers_[i]->blobs().size();j++)
			{				
				for (int k = 0; k < NGPUS; k++) 
				{
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[k]));
					ncclReduce(layers_[i]->parallel_blobs()[j*NGPUS+k]->gpu_diff(),layers_[i]->parallel_blobs()[j*NGPUS]->mutable_gpu_diff(),
															layers_[i]->parallel_blobs()[j*NGPUS]->count(),
															ncclFloat,ncclSum,0,Caffe::comms(k),NULL);	
				}							
			}			
		}	
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	}
}

template void Net<float>::BcastData();
template void Net<float>::ReduceDiff();
template void Net<double>::BcastData();
template void Net<double>::ReduceDiff();
}  // namespace caffe
