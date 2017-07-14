#include <algorithm>
#include <vector>

#include "caffe/layers/operator/cudnn_conv_layer.hpp"

namespace caffe {

#define CUDNN_STREAMS 3


template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
  ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
	
	iter_ = 0;
//----------------------------------------	
	myworkspace_.resize(1);
	myworkspace_[0] = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_]);
//----------------------------------------	


  cudnn::createFilterDesc<Dtype>(&filter_desc_,
      this->num_output_ / this->group_, this->channels_ / this->group_, this->kernel_size_, this->kernel_size_);
  if (this->layer_param_.convolution_param().bias_term()) 
  {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
   	cudnn::setTensor4dDesc<Dtype>(&bias_desc_, 1, this->num_output_/ this->group_, 1, 1); 
  } 

  cudnn::createTensor4dDesc<Dtype>(&bottom_descs_);
  cudnn::createTensor4dDesc<Dtype>(&top_descs_);
  cudnn::createConvolutionDesc<Dtype>(&conv_descs_);      
}

template <typename Dtype>
void CuDNNConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  ConvolutionLayer<Dtype>::Reshape(bottom, top);

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int height_out = top[0]->height();
  const int width_out = top[0]->width();

	if (this->group_ == 1)
	{
		cudnn::setTensor4dDesc<Dtype>(&bottom_descs_,
		    num, this->channels_, height, width);
		cudnn::setTensor4dDesc<Dtype>(&top_descs_,
		    num, this->num_output_, height_out, width_out);  
	}
	else
	{
		CHECK_EQ(this->num_output_,this->channels_);
		cudnn::setTensor4dDesc<Dtype>(&bottom_descs_,
		    1, this->channels_  / this->group_ , height, width);
		cudnn::setTensor4dDesc<Dtype>(&top_descs_,
		    1, this->num_output_  / this->group_ , height_out, width_out);  	
	}
	cudnn::setConvolutionDesc<Dtype>(&conv_descs_, 
		    this->pad_, this->pad_, this->stride_, this->stride_, this->filter_stride_, this->filter_stride_);
  //set the max work space data in case of RUNOUT of memory
  //take 448 x 448 as a exemplar
	size_t workspace_limit_bytes = 1011888000;
	if (num == 1)
		workspace_limit_bytes = 0;

  	CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(gpu_id_),
					bottom_descs_,
					filter_desc_,
					conv_descs_,
					top_descs_,
					CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
					workspace_limit_bytes,
					&fwd_algo_));			
		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(Caffe::cudnn_handle(gpu_id_),
		    bottom_descs_, 
		    top_descs_, 
		    conv_descs_, 
		    filter_desc_,
		    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		    workspace_limit_bytes, 
		    &bwd_filter_algo_) );
		CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(Caffe::cudnn_handle(gpu_id_),
		    filter_desc_, 
		    top_descs_, 
		    conv_descs_, 
		    bottom_descs_,
		    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		    workspace_limit_bytes, 
		    &bwd_data_algo_));   

            
	//fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	//bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	//bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;    


   
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			bottom_descs_,
			filter_desc_,
			conv_descs_,
			top_descs_,
			fwd_algo_,
			&(workspace_fwd_sizes_)));			   
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
		  bottom_descs_, 
		  top_descs_, 
		  conv_descs_, 
		  filter_desc_,
		  bwd_filter_algo_, 
		  &workspace_bwd_filter_sizes_));    
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			filter_desc_, 
			top_descs_, 
			conv_descs_, 
			bottom_descs_,
			bwd_data_algo_, 
			&workspace_bwd_data_sizes_) );   

  //LOG(INFO)<<" fwd_algo_ = "<<fwd_algo_ <<" "
 // 					<<" bwd_filter_algo_ ="<<bwd_filter_algo_<<" "
  //					<<" bwd_data_algo_ = "<<bwd_data_algo_;    
//-----------------------------------------------------------------------------------------	
	myworkspace_[0]->Reshape(workspace_fwd_sizes_/sizeof(Dtype)+1,1,1,1);
 	myworkspace_[0]->Reshape(workspace_bwd_data_sizes_/sizeof(Dtype)+1,1,1,1);
 	myworkspace_[0]->Reshape(workspace_bwd_filter_sizes_/sizeof(Dtype)+1,1,1,1);    
//-----------------------------------------------------------------------------------------	   
       
}
template <typename Dtype>
CuDNNConvolutionLayer<Dtype>::~CuDNNConvolutionLayer() 
{

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_descs_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_descs_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_descs_));
  
  if (this->layer_param_.convolution_param().bias_term()) 
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));

  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
}

INSTANTIATE_CLASS(CuDNNConvolutionLayer);
REGISTER_LAYER_CLASS(CuDNNConvolution);
}   // namespace caffe

