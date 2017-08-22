#include <algorithm>
#include <vector>

#include "caffe/layers/activation/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void ReLUForward(const int n, const int negative_slope, const Dtype* in, bool* flag, Dtype* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	flag[index] = in[index] > 0;
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

//template <typename Dtype>
//static __global__ void ReLUForward_test(const int n, const int negative_slope, const Dtype* in, Dtype* out) 
//{
//  CUDA_KERNEL_LOOP(index, n) 
//  {
//    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
//  }
//}

template <typename Dtype>
static __global__ void ReLUBackward(const int n, const int negative_slope, const Dtype* in_diff,const bool* flag, Dtype* out_diff) 
{
  CUDA_KERNEL_LOOP(index, n)
  {
  	if (flag[index])
    	out_diff[index] = in_diff[index];
    else
    	out_diff[index] = in_diff[index] * negative_slope;
  }
}



template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

	bool* flag_data = flag.mutable_gpu_data();
	ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	(count, negative_slope, bottom_data, flag_data,top_data);
	
	//ReLUForward_test<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	//(count, negative_slope, bottom_data,top_data);
	//CUDA_POST_KERNEL_CHECK; 
	
	//CUDA_CHECK(cudaDeviceSynchronize());  
}

template <typename Dtype>
void ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{  
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  
  
	const bool* flag_data = flag.gpu_data();
  ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, negative_slope, top_diff, flag_data, bottom_diff);     
}

template <typename Dtype>
void ReLULayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
   
 
	const bool* flag_data = flag.gpu_data();
	ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	(count, negative_slope, bottom[0]->gpu_sec_diff(), flag_data, top[0]->mutable_gpu_sec_diff());
	CUDA_POST_KERNEL_CHECK;    	
}
INSTANTIATE_LAYER_GPU_FUNCS(ReLULayer);

}  // namespace caffe
