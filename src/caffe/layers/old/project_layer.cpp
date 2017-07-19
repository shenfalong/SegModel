#include <vector>

#include "caffe/layers/project_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ProjectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void ProjectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (bottom[0]->channels() == 21)
		top[0]->Reshape(bottom[0]->num(),2,bottom[0]->height(),bottom[0]->width());
	else if (bottom[0]->channels() == 1)
		top[0]->ReshapeLike(*bottom[0]);
	else
		LOG(INFO)<<" Unsuported channels "<<bottom[0]->channels();
}

template <typename Dtype>
void ProjectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void ProjectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(ProjectLayer);
#endif

INSTANTIATE_CLASS(ProjectLayer);
REGISTER_LAYER_CLASS(Project);
}  // namespace caffe
