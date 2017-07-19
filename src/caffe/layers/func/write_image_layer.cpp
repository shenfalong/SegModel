
#include <vector>

#include "caffe/layers/func/write_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WriteImageLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
}

template <typename Dtype>
void WriteImageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}


INSTANTIATE_CLASS(WriteImageLayer);
REGISTER_LAYER_CLASS(WriteImage);
}  // namespace caffe
