#include <vector>

#include "caffe/layers/generate_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GenerateMaskLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void GenerateMaskLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GenerateMaskLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void GenerateMaskLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(GenerateMaskLayer);
#endif

INSTANTIATE_CLASS(GenerateMaskLayer);
REGISTER_LAYER_CLASS(GenerateMask);
}  // namespace caffe
		