
#include <vector>

#include "caffe/layers/data/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void NoiseLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));

	
	num_ = this->layer_param_.noise_param().num();
	channels_ = this->layer_param_.noise_param().channels();
	classes_ = this->layer_param_.noise_param().classes();
}

template <typename Dtype>
void NoiseLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (bottom.size() == 1)
	{
		CHECK_EQ(bottom.size(),1);
		CHECK_EQ(1,bottom[0]->channels());
		CHECK_EQ(num_,bottom[0]->num());
	
		top[0]->Reshape(num_,channels_+classes_,1,1);
	}
	else if (bottom.size() == 0)
	{
		if (top.size() == 1)
			top[0]->Reshape(num_,channels_,1,1);
		else
		{
			top[0]->Reshape(num_,channels_,1,1);
			top[1]->Reshape(num_,1,1,1);
		}
	}
	else 
		LOG(FATAL)<<"bottom size is wrong";
	
}

INSTANTIATE_CLASS(NoiseLayer);
REGISTER_LAYER_CLASS(Noise);
}  // namespace caffe
