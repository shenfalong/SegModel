#include <algorithm>
#include <vector>

#include "caffe/layers/operator/param_cudnn_deconv_layer.hpp"

namespace caffe {

#define CUDNN_STREAMS 3


template <typename Dtype>
void ParamCuDNNDeConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	
	num_output_ = this->layer_param_.convolution_param().num_output();
  channels_ = bottom[0]->channels();
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  pad_ = this->layer_param_.convolution_param().pad();
  stride_ = this->layer_param_.convolution_param().stride();


//----------------------------------------	
	myworkspace_.resize(1);
	myworkspace_[0] = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_]);
//----------------------------------------	


  cudnn::createFilterDesc<Dtype>(&filter_desc_, channels_,num_output_,  kernel_size_, kernel_size_);
  if (this->layer_param_.convolution_param().bias_term()) 
  {
    cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
   	cudnn::setTensor4dDesc<Dtype>(&bias_desc_, 1, this->num_output_, 1, 1); 
  } 

  cudnn::createTensor4dDesc<Dtype>(&bottom_descs_);
  cudnn::createTensor4dDesc<Dtype>(&top_descs_);
  cudnn::createConvolutionDesc<Dtype>(&conv_descs_);      
}

template <typename Dtype>
void ParamCuDNNDeConvolutionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  CHECK_EQ(bottom[1]->height(),1);
  CHECK_EQ(bottom[1]->width(),1);
	if (bottom[0]->num() > 1)
	{
		CHECK_EQ(bottom[1]->num(),bottom[0]->num());
		if (this->layer_param_.convolution_param().bias_term()) 
			CHECK_EQ(num_output_*(channels_*kernel_size_*kernel_size_+1), bottom[1]->channels());
		else
			CHECK_EQ(num_output_*channels_*kernel_size_*kernel_size_, bottom[1]->channels());
	}
	else
	{
		CHECK_EQ(bottom[1]->num(),num_output_);
		if (this->layer_param_.convolution_param().bias_term()) 
			CHECK_EQ(num_output_*(channels_*kernel_size_*kernel_size_+1), bottom[1]->num()*bottom[1]->channels());
		else
			CHECK_EQ(num_output_*channels_*kernel_size_*kernel_size_, bottom[1]->num()*bottom[1]->channels());
	}
	height_out_ = (height - 1) * stride_ + kernel_size_ - 2 * pad_;
  width_out_ = (width - 1) * stride_ + kernel_size_ - 2 * pad_;
	
	top[0]->Reshape(num,num_output_,height_out_,width_out_);

	cudnn::setTensor4dDesc<Dtype>(&bottom_descs_, 1, channels_, height, width);
	cudnn::setTensor4dDesc<Dtype>(&top_descs_, 1, num_output_, height_out_, width_out_);  
	cudnn::setConvolutionDesc<Dtype>(&conv_descs_,  pad_, pad_, stride_, stride_, 1, 1);
	
  //set the max work space data in case of RUNOUT of memory
  //take 448 x 448 as a exemplar
  size_t workspace_limit_bytes = 511888000;
	if (num == 1)
		workspace_limit_bytes = 0;
  
	CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(gpu_id_),
					top_descs_,
					filter_desc_,
					conv_descs_,
					bottom_descs_,
					CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
					workspace_limit_bytes,
					&fwd_algo_));			
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(Caffe::cudnn_handle(gpu_id_),
		  top_descs_, 
		  bottom_descs_, 
		  conv_descs_, 
		  filter_desc_,
		  CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		  workspace_limit_bytes, 
		  &bwd_filter_algo_) );
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(Caffe::cudnn_handle(gpu_id_),
		  filter_desc_, 
		  bottom_descs_, 
		  conv_descs_, 
		  top_descs_,
		  CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		  workspace_limit_bytes, 
		  &bwd_data_algo_)); 
	
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
		top_descs_,
		filter_desc_,
		conv_descs_,
		bottom_descs_,
		fwd_algo_,
		&(workspace_fwd_sizes_)));			   
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			top_descs_, 
			bottom_descs_, 
			conv_descs_, 
			filter_desc_,
			bwd_filter_algo_, 
			&workspace_bwd_filter_sizes_));    
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			filter_desc_, 
			bottom_descs_,
			conv_descs_, 
			top_descs_, 
			bwd_data_algo_, 
			&workspace_bwd_data_sizes_) );   
//-----------------------------------------------------------------------------------------	
		myworkspace_[0]->Reshape(workspace_fwd_sizes_/sizeof(Dtype)+1,1,1,1);
	 	myworkspace_[0]->Reshape(workspace_bwd_data_sizes_/sizeof(Dtype)+1,1,1,1);
	 	myworkspace_[0]->Reshape(workspace_bwd_filter_sizes_/sizeof(Dtype)+1,1,1,1);    
//-----------------------------------------------------------------------------------------	       
       
}

template <typename Dtype>
ParamCuDNNDeConvolutionLayer<Dtype>::~ParamCuDNNDeConvolutionLayer() 
{

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_descs_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_descs_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_descs_));

  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
}

INSTANTIATE_CLASS(ParamCuDNNDeConvolutionLayer);
REGISTER_LAYER_CLASS(ParamCuDNNDeConvolution);
}   // namespace caffe

