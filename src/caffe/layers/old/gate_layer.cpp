
#include <vector>

#include "caffe/layers/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GateLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<NGPUS;i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
//----------------------------------------	
	buffer_top_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_]);
//----------------------------------------	
}

template <typename Dtype>
void GateLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
//----------------------------------------		
	buffer_top_->Reshape(num,channels,height,width);
//----------------------------------------		
	//top[0]->Reshape(num,channels/4,height,width);
	top[0]->Reshape(num,channels,height,width);
}

template <typename Dtype>
void GateLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void GateLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(GateLayer);
#endif

INSTANTIATE_CLASS(GateLayer);
REGISTER_LAYER_CLASS(Gate);
}  // namespace caffe
