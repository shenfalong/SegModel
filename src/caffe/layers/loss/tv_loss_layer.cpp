
#include <vector>

#include "caffe/layers/loss/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void TVLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
}

template <typename Dtype>
void TVLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	loss_.ReshapeLike(*bottom[0]);
	top[0]->Reshape(1,1,1,1);
}


INSTANTIATE_CLASS(TVLossLayer);
REGISTER_LAYER_CLASS(TVLoss);
}  // namespace caffe
