
#include <vector>

#include "caffe/layers/activation/repmat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RepmatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
}

template <typename Dtype>
void RepmatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[1]->height();
  int width = bottom[1]->width();
//----------------------------- -------------------------------------  
  one_multiplier_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+Caffe::GPUs.size()]);
//----------------------------- -------------------------------------   
  one_multiplier_->Reshape(1,1,height,width);
	top[0]->Reshape(num,channels,height,width);
}


INSTANTIATE_CLASS(RepmatLayer);
REGISTER_LAYER_CLASS(Repmat);
}  // namespace caffe
