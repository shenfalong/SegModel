#include <set>
#include <vector>

#include "caffe/layers/random_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RandomSelectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	
	bottom_buffer_ = new Blob<Dtype>();
	sec_buffer_ = new Blob<Dtype>();
	top_buffer_ = new Blob<Dtype>();
	
	//if (this->layer_param_.sec_param().sec_feature() == SecParameter_SecFeature_SEC_0) 
	//{
	//	int num_output = this->layer_param_.convolution_param().num_output();
	//	this->layer_param_.mutable_convolution_param()->set_num_output(int(Dtype(num_output) * 0.7));
	//}	
	
	LayerParameter conv3x3_param(this->layer_param_);
  conv3x3_param.set_type("CuDNNConvolution");
  conv3x3_layer_0_ = LayerRegistry<Dtype>::CreateLayer(conv3x3_param);
  conv3x3_bottom_vec_0_.clear();
  conv3x3_bottom_vec_0_.push_back(bottom[0]);
  conv3x3_top_vec_0_.clear();
  conv3x3_top_vec_0_.push_back(bottom_buffer_);
  conv3x3_layer_0_->SetUp(conv3x3_bottom_vec_0_, conv3x3_top_vec_0_);
	
	
	stride_ = this->layer_param_.convolution_param().stride();
	
	if (this->layer_param_.sec_param().sec_feature() == SecParameter_SecFeature_SEC_0) 
	{
		LayerParameter conv3x3_param(this->layer_param_);
		conv3x3_param.set_type("CuDNNConvolution");
		conv3x3_layer_1_ = LayerRegistry<Dtype>::CreateLayer(conv3x3_param);
		conv3x3_bottom_vec_1_.clear();
		conv3x3_bottom_vec_1_.push_back(bottom[0]);
		conv3x3_top_vec_1_.clear();
		conv3x3_top_vec_1_.push_back(top_buffer_);
		conv3x3_layer_1_->SetUp(conv3x3_bottom_vec_1_, conv3x3_top_vec_1_);
	}

	if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {
  	if (this->layer_param_.sec_param().sec_feature() == SecParameter_SecFeature_SEC_0) 
  	{
		  this->blobs_.resize(1+conv3x3_layer_0_->blobs().size()+conv3x3_layer_1_->blobs().size());
		  this->lr_mult().resize(1+conv3x3_layer_0_->blobs().size()+conv3x3_layer_1_->blobs().size());
		  this->decay_mult().resize(1+conv3x3_layer_0_->blobs().size()+conv3x3_layer_1_->blobs().size());
    }
    else
    {
    	this->blobs_.resize(1+conv3x3_layer_0_->blobs().size());
		  this->lr_mult().resize(1+conv3x3_layer_0_->blobs().size());
		  this->decay_mult().resize(1+conv3x3_layer_0_->blobs().size());
    }
    
    this->blobs_[0].reset(new Blob<Dtype>(1,channels,1,1));
		for (int i=0;i<channels;i++)
			this->blobs_[0]->mutable_cpu_data()[i] = caffe_rng_rand()%4;


		this->lr_mult()[0] = 0.0;
		this->decay_mult()[0] = 0.0;
		for (int i=0;i<conv3x3_layer_0_->blobs().size();i++)
		{
			this->blobs_[1+i] = conv3x3_layer_0_->blobs()[i];		
			this->lr_mult()[1+i] = conv3x3_layer_0_->lr_mult()[i];
			this->decay_mult()[1+i] = conv3x3_layer_0_->decay_mult()[i];
		}
		if (this->layer_param_.sec_param().sec_feature() == SecParameter_SecFeature_SEC_0) 
		{
			for (int i=0;i<conv3x3_layer_1_->blobs().size();i++)
			{
				this->blobs_[1+conv3x3_layer_0_->blobs().size()+i] = conv3x3_layer_1_->blobs()[i];		
				this->lr_mult()[1+conv3x3_layer_0_->blobs().size()+i] = conv3x3_layer_1_->lr_mult()[i];
				this->decay_mult()[1+conv3x3_layer_0_->blobs().size()+i] = conv3x3_layer_1_->decay_mult()[i];
			}
		}
	}
	
}

template <typename Dtype>
void RandomSelectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	conv3x3_layer_0_->Reshape(conv3x3_bottom_vec_0_, conv3x3_top_vec_0_);
	if (this->layer_param_.sec_param().sec_feature() == SecParameter_SecFeature_SEC_0) 
	{
		conv3x3_layer_1_->Reshape(conv3x3_bottom_vec_1_, conv3x3_top_vec_1_);
	}
	bottom_buffer_->Reshape(num,channels,height/stride_,width/stride_);
	sec_buffer_->Reshape(num,channels,height/stride_,width/stride_);
	top_buffer_->Reshape(num,channels,height/stride_,width/stride_);
	
	top[0]->Reshape(num,2*channels,height/stride_,width/stride_);
}

template <typename Dtype>
void RandomSelectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void RandomSelectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

template <typename Dtype>
RandomSelectLayer<Dtype>::~RandomSelectLayer()
{
	delete top_buffer_;
}

#ifdef CPU_ONLY
STUB_GPU(RandomSelectLayer);
#endif

INSTANTIATE_CLASS(RandomSelectLayer);
REGISTER_LAYER_CLASS(RandomSelect);
}  // namespace caffe
