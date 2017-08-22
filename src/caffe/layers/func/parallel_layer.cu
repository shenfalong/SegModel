


#include <vector>

#include "caffe/layers/func/parallel_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParallelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
//-------------------------------------------------------
	if ((this->layer_param_.type() == "CuDNNBatchNorm"))
	{	
		if (Caffe::number_collect_sample != -1)
		{
			CHECK_EQ(this->parallel_blobs_.size(),4*NGPUS);
			if (Caffe::number_collect_sample == 0)
			{
				caffe_gpu_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_gpu_data());
				caffe_gpu_set(this->blobs_[3]->count(),Dtype(0),this->blobs_[3]->mutable_gpu_data());
			}		
			for (int i = 0; i < NGPUS; i++) 
			{  	
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclBcast((void *)this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[2*NGPUS+i]->count(),
		 																		ncclFloat,0,Caffe::comms(i),NULL);			
			}		
			for (int i = 0; i < NGPUS; i++) 
			{  	
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclBcast((void *)this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[3*NGPUS+i]->count(),
		 																		ncclFloat,0,Caffe::comms(i),NULL);			
			}		
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		}	 
	}
	if (this->layer_param_.type() == "BatchNorm")
	{
		if (Caffe::number_collect_sample != -1)
		{
			CHECK_EQ(this->parallel_blobs_.size(),2*NGPUS);
			if (Caffe::number_collect_sample == 0)
			{
				caffe_gpu_set(this->blobs_[0]->count(),Dtype(0),this->blobs_[0]->mutable_gpu_data());
				caffe_gpu_set(this->blobs_[1]->count(),Dtype(0),this->blobs_[1]->mutable_gpu_data());
			}		
			for (int i = 0; i < NGPUS; i++) 
			{  	
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclBcast((void *)this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[0*NGPUS+i]->count(),
		 																		ncclFloat,0,Caffe::comms(i),NULL);			
			}		
			for (int i = 0; i < NGPUS; i++) 
			{  	
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclBcast((void *)this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[1*NGPUS+i]->count(),
		 																		ncclFloat,0,Caffe::comms(i),NULL);			
			}		
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		}	 
	}
//-------------------------------------------------------
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Forward_gpu(unary_bottom_vec_[i], unary_top_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
//-------------------------------------------------------
	if ((this->layer_param_.type() == "CuDNNBatchNorm"))
	{
		if (Caffe::number_collect_sample != -1)
		{
			
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclReduce( this->parallel_blobs_[2*NGPUS+i]->gpu_data(),this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),
						this->parallel_blobs_[2*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
			}
			
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclReduce( this->parallel_blobs_[3*NGPUS+i]->gpu_data(),this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),
						this->parallel_blobs_[3*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
			}
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			caffe_gpu_scal(this->blobs_[2]->count(),Dtype(1)/Dtype(NGPUS),this->blobs_[2]->mutable_gpu_data());
			caffe_gpu_scal(this->blobs_[3]->count(),Dtype(1)/Dtype(NGPUS),this->blobs_[3]->mutable_gpu_data());	
			
		}	
	}
	if ((this->layer_param_.type() == "BatchNorm"))
	{
		if (Caffe::number_collect_sample != -1)
		{
			
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclReduce( this->parallel_blobs_[0*NGPUS+i]->gpu_data(),this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_data(),
						this->parallel_blobs_[0*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
			}
			
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclReduce( this->parallel_blobs_[1*NGPUS+i]->gpu_data(),this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_data(),
						this->parallel_blobs_[1*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
			}
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			caffe_gpu_scal(this->blobs_[0]->count(),Dtype(1)/Dtype(NGPUS),this->blobs_[0]->mutable_gpu_data());
			caffe_gpu_scal(this->blobs_[1]->count(),Dtype(1)/Dtype(NGPUS),this->blobs_[1]->mutable_gpu_data());	
			
		}	
	}
	if (this->layer_param_.type() == "BeGdLoss")
	{
		
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
			ncclReduce(this->parallel_blobs_[i]->gpu_data(),this->parallel_blobs_[i]->mutable_gpu_data(),
					this->parallel_blobs_[i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
		}
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		caffe_gpu_scal(this->blobs_[0]->count(),Dtype(1)/Dtype(NGPUS),this->blobs_[0]->mutable_gpu_data());
		
	}
//-------------------------------------------------------
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

