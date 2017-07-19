#include <vector>

#include "caffe/layers/mask_out_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MaskOutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void MaskOutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void MaskOutLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void MaskOutLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(MaskOutLayer);
#endif

INSTANTIATE_CLASS(MaskOutLayer);
REGISTER_LAYER_CLASS(MaskOut);
}  // namespace caffe
