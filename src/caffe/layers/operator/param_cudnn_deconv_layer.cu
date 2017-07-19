#ifdef USE_CUDNN
#include <vector>


#include "caffe/layers/operator/param_cudnn_deconv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParamCuDNNDeConvolutionLayer<Dtype>::Forward_gpu( const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int group = group_;
	
	for (int n=0;n<num;n++)
	{
		CUDNN_CHECK(cudnnConvolutionBackwardData(
			  Caffe::cudnn_handle(gpu_id_),
			  cudnn::dataType<Dtype>::one,
			  filter_desc_, bottom[1]->gpu_data()+bottom[1]->offset(n),
			  bottom_descs_, bottom[0]->gpu_data()+bottom[0]->offset(n),
			  conv_descs_,
			  bwd_data_algo_, work_space1, workspace_bwd_data_sizes_,
			  cudnn::dataType<Dtype>::zero,
			  top_descs_, top[0]->mutable_gpu_data()+top[0]->offset(n)));  
		if (this->layer_param_.convolution_param().bias_term()) 
		{
			CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(gpu_id_),
					  cudnn::dataType<Dtype>::one,
					  bias_desc_, bottom[1]->gpu_data()+bottom[1]->offset(n)+num_output_ * channels_ *kernel_size_*kernel_size_,
					  cudnn::dataType<Dtype>::one,
					  top_descs_, top[0]->mutable_gpu_data()+top[0]->offset(n)));
		}                
	}		 
}

template <typename Dtype>
void ParamCuDNNDeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	
	for (int n=0;n<num;n++)
	{
		if (this->layer_param_.convolution_param().bias_term())
		{
			CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(gpu_id_),
				      cudnn::dataType<Dtype>::one,
				      top_descs_,  top[0]->gpu_diff()+top[0]->offset(n),
				      cudnn::dataType<Dtype>::zero,
				      bias_desc_, bottom[1]->mutable_gpu_diff()+bottom[1]->offset(n)+num_output_ * channels_ *kernel_size_*kernel_size_));
		}  
		if (this->has_bottom_sec_diff_ ==  false)
		{             
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(gpu_id_),
					cudnn::dataType<Dtype>::one,
					top_descs_, top[0]->gpu_diff()+top[0]->offset(n),
					bottom_descs_,    bottom[0]->gpu_data()+bottom[0]->offset(n),
					conv_descs_,
					bwd_filter_algo_, work_space0, workspace_bwd_filter_sizes_,
					cudnn::dataType<Dtype>::zero,
					filter_desc_, bottom[1]->mutable_gpu_diff()+bottom[1]->offset(n)));
		}
		else
		{
			CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(gpu_id_),
					cudnn::dataType<Dtype>::one,
					top_descs_, top[0]->gpu_diff()+top[0]->offset(n),
					bottom_descs_,    bottom[0]->gpu_data()+bottom[0]->offset(n),
					conv_descs_,
					bwd_filter_algo_, work_space0, workspace_bwd_filter_sizes_,
					cudnn::dataType<Dtype>::one,
					filter_desc_, bottom[1]->mutable_gpu_diff()+bottom[1]->offset(n)));
		}
		CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
			  cudnn::dataType<Dtype>::one,
			  top_descs_, top[0]->gpu_diff()+top[0]->offset(n),
			  filter_desc_, bottom[1]->gpu_data()+bottom[1]->offset(n),
			  conv_descs_,
			  fwd_algo_, work_space0, workspace_fwd_sizes_,
			  cudnn::dataType<Dtype>::zero,
			  bottom_descs_, bottom[0]->mutable_gpu_diff()+bottom[0]->offset(n)));   
	}
	this->has_bottom_sec_diff_ = false;
}
template <typename Dtype>
void ParamCuDNNDeConvolutionLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	
	for (int n=0;n<num;n++)
	{
		CUDNN_CHECK(cudnnConvolutionBackwardData(
				    Caffe::cudnn_handle(gpu_id_),
				    cudnn::dataType<Dtype>::one,
				    filter_desc_, bottom[1]->gpu_data()+bottom[1]->offset(n),
				    bottom_descs_, bottom[0]->gpu_sec_diff()+bottom[0]->offset(n),
				    conv_descs_,
				    bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
				    cudnn::dataType<Dtype>::zero,
				    top_descs_, top[0]->mutable_gpu_sec_diff()+top[0]->offset(n))); 		   
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
				Caffe::cudnn_handle(gpu_id_),
				cudnn::dataType<Dtype>::one,
				top_descs_,    top[0]->gpu_diff()+top[0]->offset(n),
				bottom_descs_, bottom[0]->gpu_sec_diff()+bottom[0]->offset(n),
				conv_descs_,
				bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
				cudnn::dataType<Dtype>::one,
				filter_desc_, bottom[1]->mutable_gpu_diff()+bottom[1]->offset(n))); 
	}
	this->has_bottom_sec_diff_ = true;  
}

INSTANTIATE_LAYER_GPU_FUNCS(ParamCuDNNDeConvolutionLayer);

}  // namespace caffe
#endif
