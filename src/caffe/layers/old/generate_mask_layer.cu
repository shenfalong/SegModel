#include <vector>

#include "caffe/layers/generate_mask_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
static __global__ void mask_kernel(int count,int height, int width, const Dtype * in, Dtype * out)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height ;
  	int h = index / width % height;
  	int w = index % width;
  	int cur_label = in[index];
  	bool is_edge = false;
  	for (int ih = max(h-1,0);ih<min(h+2,height);ih++)
  	for (int iw = max(w-1,0);iw<min(w+2,width);iw++)
  	{
  		int neighbor_index = (n* height + ih)*width + iw;
  		if (in[neighbor_index] != cur_label)
  		{
  			is_edge = true;
  			break;
  		}
		}
		if (is_edge)
			out[index] = 1;
  	
  }
}    


template <typename Dtype>
void GenerateMaskLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  caffe_gpu_set(top[0]->count(),Dtype(0),top[0]->mutable_gpu_data());
  
  mask_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
   (bottom[0]->count(),height,width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}

template <typename Dtype>
void GenerateMaskLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
}

INSTANTIATE_LAYER_GPU_FUNCS(GenerateMaskLayer);
}  // namespace caffe
		
