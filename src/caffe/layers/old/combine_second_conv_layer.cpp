
#include <vector>

#include "caffe/layers/combine_second_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CombineSecConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	
//----------------------------- work space ------------------------- 
	// convolution layer uses workspace, pay attention to the conflict of blobs
	sec_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+1*Caffe::GPUs.size()]);
	first_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+2*Caffe::GPUs.size()]);
	second_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+3*Caffe::GPUs.size()]);
	bottom_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+4*Caffe::GPUs.size()]);
//----------------------------- -------------------------------------	
	//sec_buffer_ = new Blob<Dtype>();
	//first_buffer_ = new Blob<Dtype>();
	//second_buffer_ = new Blob<Dtype>();
	//bottom_buffer_ = new Blob<Dtype>();
	
	LayerParameter conv1_param(this->layer_param_);//bottom[0] --> first_buffer_	
  conv1_param.set_type("CuDNNConvolution");
  conv1_layer_ = LayerRegistry<Dtype>::CreateLayer(conv1_param);
  conv1_bottom_vec_.clear();
  conv1_bottom_vec_.push_back(bottom[0]);
  conv1_top_vec_.clear();
  conv1_top_vec_.push_back(first_buffer_);
  conv1_layer_->SetUp(conv1_bottom_vec_, conv1_top_vec_);
	
	int kernel_num = conv1_param.convolution_param().num_output();
	LayerParameter conv2_param(this->layer_param_);//bottom[0] --> sec_buffer_
  conv2_param.set_type("CuDNNConvolution");
  conv2_param.mutable_convolution_param()->set_num_output(kernel_num*8);
  conv2_layer_ = LayerRegistry<Dtype>::CreateLayer(conv2_param);
  conv2_bottom_vec_.clear();
  conv2_bottom_vec_.push_back(bottom[0]);
  conv2_top_vec_.clear();
  conv2_top_vec_.push_back(sec_buffer_);
  conv2_layer_->SetUp(conv2_bottom_vec_, conv2_top_vec_);
  
  
  LayerParameter sec_param(this->layer_param_);//sec_buffer_, bottom[0] --> second_buffer_
  sec_param.set_type("SecConv");
  sec_layer_ = LayerRegistry<Dtype>::CreateLayer(sec_param);
  sec_bottom_vec_.clear();
  sec_bottom_vec_.push_back(sec_buffer_);
  sec_bottom_vec_.push_back(bottom[0]);
  sec_top_vec_.clear();
  sec_top_vec_.push_back(second_buffer_);
  sec_layer_->SetUp(sec_bottom_vec_, sec_top_vec_);
  
  LayerParameter concat_param(this->layer_param_);//first_buffer_, second_buffer_ --> top[0]
  concat_param.set_type("Concat");
  concat_layer_ = LayerRegistry<Dtype>::CreateLayer(concat_param);
  concat_bottom_vec_.clear();
  concat_bottom_vec_.push_back(first_buffer_);
  concat_bottom_vec_.push_back(second_buffer_);
  concat_top_vec_.clear();
  concat_top_vec_.push_back(top[0]);
  concat_layer_->SetUp(concat_bottom_vec_, concat_top_vec_);
  
  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {

  	this->blobs_.resize(conv1_layer_->blobs().size()+conv2_layer_->blobs().size());
	  this->lr_mult().resize(conv1_layer_->blobs().size()+conv2_layer_->blobs().size());
	  this->decay_mult().resize(conv1_layer_->blobs().size()+conv2_layer_->blobs().size());

	  for (int i=0; i<conv1_layer_->blobs().size(); i++)
	  {
			this->blobs_[i] = conv1_layer_->blobs()[i];		
			this->lr_mult()[i] = conv1_layer_->lr_mult()[i];
			this->decay_mult()[i] = conv1_layer_->decay_mult()[i];
		}
		for (int i=0; i<conv2_layer_->blobs().size(); i++)
	  {
			this->blobs_[conv1_layer_->blobs().size()+i] = conv2_layer_->blobs()[i];		
			this->lr_mult()[conv1_layer_->blobs().size()+i] = conv2_layer_->lr_mult()[i];
			this->decay_mult()[conv1_layer_->blobs().size()+i] = conv2_layer_->decay_mult()[i];
		}
  }
}

template <typename Dtype>
void CombineSecConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
//----------------------------- work space ------------------------- 
	// convolution layer uses workspace, pay attention to the conflict of blobs
	sec_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+1*Caffe::GPUs.size()]);
	first_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+2*Caffe::GPUs.size()]);
	second_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+3*Caffe::GPUs.size()]);
	bottom_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+4*Caffe::GPUs.size()]);
//----------------------------- -------------------------------------	
	bottom_buffer_->ReshapeLike(*bottom[0]);
	
	conv1_layer_->Reshape(conv1_bottom_vec_, conv1_top_vec_);
	conv2_layer_->Reshape(conv2_bottom_vec_, conv2_top_vec_);
	sec_layer_->Reshape(sec_bottom_vec_, sec_top_vec_);
	concat_layer_->Reshape(concat_bottom_vec_, concat_top_vec_);
}

template <typename Dtype>
void CombineSecConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void CombineSecConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(CombineSecConvLayer);
#endif

INSTANTIATE_CLASS(CombineSecConvLayer);
REGISTER_LAYER_CLASS(CombineSecConv);
}  // namespace caffe
