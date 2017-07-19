#include <vector>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/parallel_collect_layer.hpp"

namespace caffe 
{
template <typename Dtype>
void ParallelCollectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,const vector<Blob<Dtype>*>& top)
{
  
}

template <typename Dtype>
void ParallelCollectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int num = 0;
	for(int i=0;i<bottom.size();i++)
		num += bottom[i]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
  top[0]->Reshape(num,channels,height,width);
}

template <typename Dtype>
void ParallelCollectLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void ParallelCollectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(ParallelCollectLayer);
#endif  

INSTANTIATE_CLASS(ParallelCollectLayer);
REGISTER_LAYER_CLASS(ParallelCollect);
}  // namespace caffe
