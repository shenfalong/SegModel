// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/operator/cudnn_deconv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void sync_deconv_groups() { }

template <typename Dtype>
void CuDNNDeConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*> &top) 
{
	
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
 	if (this->group_ == 1)
 	{

	 	CUDNN_CHECK(cudnnConvolutionBackwardData(
		      Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      filter_desc_, this->blobs_[0]->gpu_data(),
		      bottom_descs_, bottom_data,
		      conv_descs_,
		      bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
		      cudnn::dataType<Dtype>::zero,
		      top_descs_, top_data)); 	
		if (this->layer_param_.convolution_param().bias_term()) 
		{
		  const Dtype* bias_data = this->blobs_[1]->gpu_data();
		  CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(gpu_id_),
		        cudnn::dataType<Dtype>::one,
		        bias_desc_, bias_data,
		        cudnn::dataType<Dtype>::one,
		        top_descs_, top_data));
		}           
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		int group = this->group_;
		
		int height_out = top[0]->height();
		int width_out = top[0]->width();
		
		for (int n = 0; n < num; n++) 
	 	{
			for (int g = 0; g < group; g++) 
			{
				CUDNN_CHECK(cudnnConvolutionBackwardData(
				  Caffe::cudnn_handle(gpu_id_),
				  cudnn::dataType<Dtype>::one,
				  filter_desc_, this->blobs_[0]->gpu_data() + g*channels / group*channels / group*this->blobs_[0]->height()*this->blobs_[0]->width(),
				  bottom_descs_, bottom_data + bottom[0]->offset(n) + g*channels / group*height*width,
				  conv_descs_,
				  bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
				  cudnn::dataType<Dtype>::zero,
				  top_descs_, top_data + top[0]->offset(n) + g*channels / group*height_out*width_out)); 		
			}
			if (this->layer_param_.convolution_param().bias_term()) 
			{
				for (int g = 0; g < this->group_; g++) 
				{
					CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(gpu_id_),
								cudnn::dataType<Dtype>::one,
								bias_desc_, this->blobs_[1]->gpu_data() + g * channels / group,
								cudnn::dataType<Dtype>::one,
								top_descs_, top[0]->mutable_gpu_data() + top[0]->offset(n) + g*channels / group*height_out*width_out));
				}
			}      
		}	    
	}	
}

template <typename Dtype>
void CuDNNDeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* top_data = top[0]->gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
		
  if (this->group_ == 1)
 	{
 		if (this->layer_param_.convolution_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
		{
		  CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(gpu_id_),
		        cudnn::dataType<Dtype>::one,
		        top_descs_,  top_diff,
		        cudnn::dataType<Dtype>::one,
		        bias_desc_, this->blobs_[1]->mutable_gpu_diff()));
		}
		CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      top_descs_, top_diff,
		      filter_desc_, weight,
		      conv_descs_,
		      fwd_algo_, work_space0, workspace_fwd_sizes_,
		      cudnn::dataType<Dtype>::zero,
		      bottom_descs_, bottom_diff));   
		if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
		{          
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(
		      Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      top_descs_,    top_diff,
		      bottom_descs_, bottom_data,
		      conv_descs_,
		      bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
		      cudnn::dataType<Dtype>::one,
		      filter_desc_, this->blobs_[0]->mutable_gpu_diff()));           
		}    
	}
	else
	{	
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		int group = this->group_;
		
		int height_out = top[0]->height();
		int width_out = top[0]->width();
		
		for (int n = 0; n < num; n++) 
	 	{
	 		if (this->layer_param_.convolution_param().bias_term()) 
			{				
				for (int g = 0; g < group; g++) 
				{
					CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(gpu_id_),
							  cudnn::dataType<Dtype>::one,
							  top_descs_,  top[0]->gpu_diff() + top[0]->offset(n) + g*channels / group*height_out*width_out,
							  cudnn::dataType<Dtype>::one,
							  bias_desc_, this->blobs_[1]->mutable_gpu_diff() + g * channels / group));
				}
			}
			for (int g = 0; g < group; g++) 
			{
				CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      top_descs_, top_diff + top[0]->offset(n) + g*channels / group*height_out*width_out,
		      filter_desc_, this->blobs_[0]->gpu_data() + g *channels / group*channels / group* this->blobs_[0]->height()*this->blobs_[0]->width(),
		      conv_descs_,
		      fwd_algo_, work_space0, workspace_fwd_sizes_,
		      cudnn::dataType<Dtype>::zero,
		      bottom_descs_, bottom_diff + bottom[0]->offset(n) + g*channels / group*height*width));
			}
			for (int g = 0; g < group; g++) 
			{
				CUDNN_CHECK(cudnnConvolutionBackwardFilter(
		      Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      top_descs_,    top_diff + top[0]->offset(n) + g *channels / group* height_out*width_out,
		      bottom_descs_, bottom_data + bottom[0]->offset(n) + g *channels / group* height*width,
		      conv_descs_,
		      bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
		      cudnn::dataType<Dtype>::one,
		      filter_desc_, this->blobs_[0]->mutable_gpu_diff() + g *channels / group*channels / group* this->blobs_[0]->height()*this->blobs_[0]->width()));   
			}   
		}	
	}	     
}
template <typename Dtype>
void CuDNNDeConvolutionLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*> &top) 
{
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	CUDNN_CHECK(cudnnConvolutionBackwardData(
		      Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType<Dtype>::one,
		      filter_desc_, this->blobs_[0]->gpu_data(),
		      bottom_descs_, bottom[0]->gpu_sec_diff(),
		      conv_descs_,
		      bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
		      cudnn::dataType<Dtype>::zero,
		      top_descs_, top[0]->mutable_gpu_sec_diff())); 		   

  if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{     
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
			  Caffe::cudnn_handle(gpu_id_),
			  cudnn::dataType<Dtype>::one,
			  top_descs_,    top[0]->gpu_diff(),
			  bottom_descs_, bottom[0]->gpu_sec_diff(),
			  conv_descs_,
			  bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
			  cudnn::dataType<Dtype>::one,
			  filter_desc_, this->blobs_[0]->mutable_gpu_diff()));   
	}

}
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNDeConvolutionLayer);

}  // namespace caffe
