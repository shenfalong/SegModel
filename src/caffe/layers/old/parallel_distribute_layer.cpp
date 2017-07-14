#include <vector>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/parallel_distribute_layer.hpp"

namespace caffe 
{
template <typename Dtype>
void ParallelDistributeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}
template <typename Dtype>
void ParallelDistributeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int num = bottom[0]->num() / top.size();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	for(int i=0;i<top.size();i++) 	
		top[i]->Reshape(num,channels,height,width); 
}

template <typename Dtype>
void ParallelDistributeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void ParallelDistributeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}
#ifdef CPU_ONLY
STUB_GPU(ParallelDistributeLayer);
#endif  

INSTANTIATE_CLASS(ParallelDistributeLayer);
REGISTER_LAYER_CLASS(ParallelDistribute);
}  // namespace caffe
