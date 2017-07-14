// Copyright 2013 Yangqing Jia

#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using std::vector;

namespace caffe {

template <typename Dtype>
class Layer
{
 public:
  explicit Layer(const LayerParameter& param): layer_param_(param) 
  {
  	lr_mult_.clear();
		decay_mult_.clear();
  	for(int i=0;i<param.param_size();i++)
  	{
  		lr_mult_.push_back(param.param(i).lr_mult());
			decay_mult_.push_back(param.param(i).decay_mult());
  	}
		has_bottom_sec_diff_ = false;
  }
  virtual ~Layer(){};

  
  void SetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)  
  {     
  	LayerSetUp(bottom, top); 
  	Reshape(bottom, top); 
  }
  
  inline Dtype Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  inline void SecForward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  inline void Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);


  vector<shared_ptr<Blob<Dtype> > >& blobs() { return blobs_; }
  vector<shared_ptr<Blob<Dtype> > >& first_moment() { return first_moment_; }
  vector<shared_ptr<Blob<Dtype> > >& second_moment() { return second_moment_; }
  vector<shared_ptr<Blob<Dtype> > >& parallel_blobs() { return parallel_blobs_; }
	vector<Dtype>& lr_mult() { return lr_mult_; }
	vector<Dtype>& decay_mult() { return decay_mult_; }
	

  const LayerParameter& layer_param() { return layer_param_; }
  virtual void ToProto(LayerParameter* param, bool write_diff = false);
	virtual void compute_sec_loss(const vector<Blob<Dtype>*>& top, const Dtype sec_loss_weight);
	
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) = 0;
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) = 0;
	
 protected:
  LayerParameter layer_param_;
  vector<Dtype> lr_mult_;
	vector<Dtype> decay_mult_;
  vector<shared_ptr<Blob<Dtype> > > blobs_;
  vector<shared_ptr<Blob<Dtype> > > first_moment_;
  vector<shared_ptr<Blob<Dtype> > > second_moment_;
  
  vector<shared_ptr<Blob<Dtype> > > parallel_blobs_;
	bool has_bottom_sec_diff_;
	
  DISABLE_COPY_AND_ASSIGN(Layer);
};  

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();

	CUDA_SYNCHRONIZE;
	Forward_gpu(bottom, top);
	CUDA_SYNCHRONIZE;
	
	Dtype loss_weight = layer_param_.include().loss_weight();
	
	Dtype loss = 0;
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());
		for (int i=0;i<top.size();i++)
			loss += top[i]->cpu_data()[0] * loss_weight / Dtype(top.size());
	}

	return loss;
};


template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	Dtype loss_weight = layer_param_.include().loss_weight();
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());	
		for (int i=0;i<top.size();i++)
			top[i]->mutable_cpu_diff()[0] = loss_weight / Dtype(top.size());
	}

	CUDA_SYNCHRONIZE;
  Backward_gpu(top, bottom);	
 	CUDA_SYNCHRONIZE;

};

template <typename Dtype>
inline void Layer<Dtype>::SecForward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();
	Dtype sec_loss_weight = layer_param_.include().sec_loss_weight();
	
	if (sec_loss_weight > 0 && Caffe::second_pass())
	{
		compute_sec_loss(top,sec_loss_weight / Dtype(top.size()));
	}
	else
	{
		CUDA_SYNCHRONIZE;
		SecForward_gpu(bottom, top);
		CUDA_SYNCHRONIZE;
	}
};

//---------------------------------------------------------------------------------------
}  // namespace caffe

#endif  // CAFFE_LAYER_H_
