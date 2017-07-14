#include <vector>

#include "caffe/layers/func/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
    

template <typename Dtype>
void SilenceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
}

template <typename Dtype>
void SilenceLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{

		
}
template <typename Dtype>
void SilenceLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
}
INSTANTIATE_LAYER_GPU_FUNCS(SilenceLayer);
}  // namespace caffe
		
