#include <string>
#include <vector>
#include <boost/thread.hpp>

#include "caffe/layers/data/base_data_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void BaseDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
  data_transformer_.reset( new DataTransformer<Dtype>(transform_param_));
  
	DataLayerSetUp(bottom, top);
	

//--------------------------------------------------		

  prefetch_data_.cpu_data();
	prefetch_label_.cpu_data();
//--------------------------------------------------	

	
  thread_.reset(new boost::thread(&BaseDataLayer<Dtype>::InternalThreadEntry,this,gpu_id_));
}

template <typename Dtype>
void BaseDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	top[0]->Reshape(prefetch_data_.num(),prefetch_data_.channels(),
																					prefetch_data_.height(),prefetch_data_.width());
																					
	top[1]->Reshape(prefetch_label_.num(),prefetch_label_.channels(),
																				prefetch_label_.height(),prefetch_label_.width());																																		
}

template <typename Dtype>
BaseDataLayer<Dtype>::~BaseDataLayer()
{
	if (transformed_data_.count() > 0)
		transformed_data_.set_cpu_data(NULL);
	if (transformed_label_.count() > 0)
		transformed_label_.set_cpu_data(NULL);
}

INSTANTIATE_CLASS(BaseDataLayer);
}  // namespace caffe
