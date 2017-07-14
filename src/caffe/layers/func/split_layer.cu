#include <vector>

#include "caffe/layers/func/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SplitLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

	for (int i = 1; i < top.size(); ++i)
		caffe_copy(top[i]->count(),bottom[0]->gpu_data(),top[i]->mutable_gpu_data());

}

template <typename Dtype>
void SplitLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
  
  caffe_gpu_add(bottom[0]->count(), Dtype(1), top[0]->gpu_diff(), Dtype(1),top[1]->gpu_diff(),  
  										bottom[0]->mutable_gpu_diff());
  for (int i = 2; i < top.size(); ++i) 
  {
    const Dtype* top_diff = top[i]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy(bottom[0]->count(), Dtype(1.), top_diff, bottom_diff);
  }
}
template <typename Dtype>
void SplitLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	for (int i = 1; i < top.size(); ++i)
			caffe_copy(top[i]->count(),bottom[0]->gpu_sec_diff(),top[i]->mutable_gpu_sec_diff());
}

INSTANTIATE_LAYER_GPU_FUNCS(SplitLayer);

}  // namespace caffe]);
