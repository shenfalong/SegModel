#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/operator/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{ 
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	const Dtype* bottom_data = bottom[0]->gpu_data();
	Dtype* top_data = top[0]->mutable_gpu_data();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, num_output, channels*height*width, 
												(Dtype)1., bottom_data, weight, 
												(Dtype)0., top_data);
												
	if (this->layer_param_.inner_product_param().bias_term())
	{
	  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, num_output, 1, 
	  										(Dtype)1., bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(), 
	  										(Dtype)1., top_data);
	}  				
				 
}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
  int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width(); 
     
  const Dtype* top_diff = top[0]->gpu_diff();
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype * bottom_diff = bottom[0]->mutable_gpu_diff();
  
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num, channels*height*width, num_output, 
											(Dtype)1., top_diff, this->blobs_[0]->gpu_data(), 
											(Dtype)0., bottom_diff);
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output, channels*height*width, num, 
													(Dtype)1., top_diff, bottom_data, 
													(Dtype)1., this->blobs_[0]->mutable_gpu_diff());
	}
  if (this->layer_param_.inner_product_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
  {
    const Dtype* top_diff = top[0]->gpu_diff();
    
    caffe_gpu_gemv<Dtype>(CblasTrans, num, num_output, 
    											(Dtype)1., top_diff, bias_multiplier_.gpu_data(), 
    											(Dtype)1., this->blobs_[1]->mutable_gpu_diff());
  }
}
template <typename Dtype>
void InnerProductLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{ 
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, num, num_output, channels*height*width, 
											(Dtype)1., bottom[0]->gpu_sec_diff(), this->blobs_[0]->gpu_data(), 
											(Dtype)0., top[0]->mutable_gpu_sec_diff());	
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{	
		caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, num_output, channels*height*width, num, 
												(Dtype)1., top[0]->gpu_diff(), bottom[0]->gpu_sec_diff(),
												(Dtype)1., this->blobs_[0]->mutable_gpu_diff());
	}							
}
INSTANTIATE_LAYER_GPU_FUNCS(InnerProductLayer);

}  // namespace caffe
