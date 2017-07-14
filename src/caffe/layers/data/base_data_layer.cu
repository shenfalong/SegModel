#include <vector>
#include "caffe/solver.hpp"
#include "caffe/layers/data/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BaseDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
 	//if (data_iter_ == Solver<Dtype>::iter())
	//	return;
	//data_iter_++;
		
		
  // wait until the end of the thread
  if (thread_.get() != NULL && thread_->joinable())
  	thread_->join(); 
	


	caffe_copy(top[0]->count(), prefetch_data_.cpu_data() , top[0]->mutable_gpu_data());
	
	caffe_copy(top[1]->count(), prefetch_label_.cpu_data(),top[1]->mutable_gpu_data());

 
#if 0
FILE *fid = fopen("debug","wb");
Dtype height = top[0]->height();
Dtype width = top[0]->width();

LOG(INFO)<<" height = "<<height<<", width "<<width;
fwrite(top[0]->cpu_data(),sizeof(Dtype), top[0]->count(),fid);
fclose(fid);
LOG(FATAL)<<"----------------------------";
#endif
  thread_.reset(new boost::thread(&BaseDataLayer<Dtype>::InternalThreadEntry,this,gpu_id_));
}

template <typename Dtype>
void BaseDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}
template <typename Dtype>
void BaseDataLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(BaseDataLayer);
}  // namespace caffe
