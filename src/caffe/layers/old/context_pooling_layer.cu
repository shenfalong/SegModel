#include <cfloat>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/context_pooling_layer.hpp"
#include "caffe/proto/caffe.pb.h"


using std::max;
using std::min;

namespace caffe 
{
//************************************* around ****************************************
template <typename Dtype>
static __global__ void ContextPoolForwardMAXAROUND_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* argmax_data) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph - context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph + context_height, height-1);
    int wend = min(pw + context_width, width-1);
 
    bool is_empty = (hend < hstart) || (wend < wstart);
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    int maxidx = -1;
   
    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
        int bottom_index = h * width + w;
        if (temp_bottom_data[bottom_index] > maxval)
        {
          maxval = temp_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      } 
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}
template <typename Dtype>
static __global__ void ContextPoolForwardAVEAROUND_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* arg_count) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph - context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph + context_height, height-1);
    int wend = min(pw + context_width, width-1);

    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    Dtype sum = 0;
    int count = 0;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
      	count ++;
        sum += temp_bottom_data[h * width + w];
      }  
    arg_count[index] = count;
    top_data[index] = sum / count;
  }
}
template <typename Dtype>
static __global__ void ContextPoolBackwardMAXAROUND_kernel(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
		const Dtype* offset_top_diff = top_diff + offset;
		const int* offset_argmax_data = argmax_data + offset;
		
		int hstart = max(h - context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h + context_height, height-1);
    int wend = min(w + context_width, width-1);


		Dtype gradient = 0;
		for (int ph = hstart; ph <= hend; ++ph)
		for (int pw = wstart; pw <= wend; ++pw)
			if (offset_argmax_data[ph * width + pw] == (h * width + w))       
				gradient += offset_top_diff[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
template <typename Dtype>
static __global__ void ContextPoolBackwardAVEAROUND_kernel(const int nthreads, const Dtype* top_diff,
    const int* arg_count, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
    const int* offset_arg_count = arg_count + offset;
    const Dtype* offset_top_diff = top_diff + offset;
		
		int hstart = max(h - context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h + context_height, height-1);
    int wend = min(w + context_width, width-1);


		Dtype gradient = 0;
    for (int ph = hstart; ph <= hend; ++ph)
      for (int pw = wstart; pw <= wend; ++pw) 
        gradient += offset_top_diff[ph * width + pw] / offset_arg_count[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
//************************************** top ********************
template <typename Dtype>
static __global__ void ContextPoolForwardMAXTOP_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* argmax_data) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph + 1*context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph + 3*context_height, height-1);
    int wend = min(pw + context_width, width-1);
 
    bool is_empty = hstart > height-1;
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    int maxidx = -1;
   
    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
        int bottom_index = h * width + w;
        if (temp_bottom_data[bottom_index] > maxval)
        {
          maxval = temp_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      } 
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}
template <typename Dtype>
static __global__ void ContextPoolForwardAVETOP_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* arg_count) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph + 1*context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph + 3*context_height, height-1);
    int wend = min(pw + context_width, width-1);

		bool is_empty = hstart > height-1;

    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    Dtype sum = 0;
    int count = 0;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
      	count ++;
        sum += temp_bottom_data[h * width + w];
      }  
    arg_count[index] = count;
    top_data[index] = is_empty ? 0 : sum / count;
  }
}
template <typename Dtype>
static __global__ void ContextPoolBackwardMAXTOP_kernel(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
		const Dtype* offset_top_diff = top_diff + offset;
		const int* offset_argmax_data = argmax_data + offset;
		
		int hstart = max(h -3*context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h -1*context_height, height-1);
    int wend = min(w + context_width, width-1);
			
		Dtype gradient = 0;
		for (int ph = hstart; ph <= hend; ++ph)
		for (int pw = wstart; pw <= wend; ++pw)
			if (offset_argmax_data[ph * width + pw] == (h * width + w))       
				gradient += offset_top_diff[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
template <typename Dtype>
static __global__ void ContextPoolBackwardAVETOP_kernel(const int nthreads, const Dtype* top_diff,
    const int* arg_count, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
    const int* offset_arg_count = arg_count + offset;
    const Dtype* offset_top_diff = top_diff + offset;
		
		int hstart = max(h - 3*context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h - 1*context_height, height-1);
    int wend = min(w + context_width, width-1);


		Dtype gradient = 0;
    for (int ph = hstart; ph <= hend; ++ph)
      for (int pw = wstart; pw <= wend; ++pw) 
        gradient += offset_top_diff[ph * width + pw] / offset_arg_count[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
//*************************************** bottom ********************
template <typename Dtype>
static __global__ void ContextPoolForwardMAXBOTTOM_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* argmax_data) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph - 3*context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph - 1*context_height, height-1);
    int wend = min(pw + context_width, width-1);
 
    bool is_empty = hend < 0;
    
    Dtype maxval = is_empty ? 0 : -FLT_MAX;
    int maxidx = -1;
   
    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
        int bottom_index = h * width + w;
        if (temp_bottom_data[bottom_index] > maxval)
        {
          maxval = temp_bottom_data[bottom_index];
          maxidx = bottom_index;
        }
      } 
    top_data[index] = maxval;
    argmax_data[index] = maxidx;
  }
}
template <typename Dtype>
static __global__ void ContextPoolForwardAVEBOTTOM_kernel(const int nthreads, const Dtype* bottom_data,
     const int channels, const int height,const int width, const int context_height,
    const int context_width, Dtype* top_data, int* arg_count) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % width;
    int ph = (index / width) % height;
    int c = (index / width / height) % channels;
    int n = index / width / height / channels;

    int hstart = max(ph - 3*context_height,0);
    int wstart = max(pw - context_width, 0);
    int hend = min(ph - 1*context_height, height-1);
    int wend = min(pw + context_width, width-1);

		bool is_empty = hend < 0;

    const Dtype * temp_bottom_data =  bottom_data + (n * channels + c) * height * width;
    Dtype sum = 0;
    int count = 0;
    for (int h = hstart; h <= hend; ++h)
      for (int w = wstart; w <= wend; ++w)
      {
      	count ++;
        sum += temp_bottom_data[h * width + w];
      }  
    arg_count[index] = count;
    top_data[index] = is_empty ? 0 : sum / count;
  }
}
template <typename Dtype>
static __global__ void ContextPoolBackwardMAXBOTTOM_kernel(const int nthreads, const Dtype* top_diff,
    const int* argmax_data, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
		const Dtype* offset_top_diff = top_diff + offset;
		const int* offset_argmax_data = argmax_data + offset;
		
		int hstart = max(h + 1*context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h + 3*context_height, height-1);
    int wend = min(w + context_width, width-1);


		Dtype gradient = 0;
		for (int ph = hstart; ph <= hend; ++ph)
		for (int pw = wstart; pw <= wend; ++pw)
			if (offset_argmax_data[ph * width + pw] == (h * width + w))       
				gradient += offset_top_diff[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
template <typename Dtype>
static __global__ void ContextPoolBackwardAVEBOTTOM_kernel(const int nthreads, const Dtype* top_diff,
    const int* arg_count, const int channels, const int height, const int width,
    const int context_height, const int context_width, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index,nthreads)
  {
		// (n, c, h, w) coords in bottom data
		int w = index % width;
		int h = (index / width) % height;
		int c = (index / width / height) % channels;
		int n = index / width / height / channels;



    int offset =  (n * channels + c) * height * width;
    const int* offset_arg_count = arg_count + offset;
    const Dtype* offset_top_diff = top_diff + offset;
		
		int hstart = max(h + 1*context_height,0);
    int wstart = max(w - context_width, 0);
    int hend = min(h + 3*context_height, height-1);
    int wend = min(w + context_width, width-1);


		Dtype gradient = 0;
    for (int ph = hstart; ph <= hend; ++ph)
      for (int pw = wstart; pw <= wend; ++pw) 
        gradient += offset_top_diff[ph * width + pw] / offset_arg_count[ph * width + pw];
				    

		bottom_diff[index] = gradient;
 	}
}
//*********************************************************************************************************
template <typename Dtype>
void ContextPoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) 
{
	const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int* argmax_data = max_idx_.mutable_gpu_data();
  int* arg_count = arg_count_.mutable_gpu_data();

		if (this->layer_param_.context_pooling_param().mode() == "around")
		{
			switch (this->layer_param_.context_pooling_param().pool())
			{
				case ContextPoolingParameter_PoolMethod_MAX:
				  ContextPoolForwardMAXAROUND_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, argmax_data);
				  break;
				case ContextPoolingParameter_PoolMethod_AVE:
				  ContextPoolForwardAVEAROUND_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, arg_count);
			}
		}
		else if (this->layer_param_.context_pooling_param().mode() == "top")
		{
		  case ContextPoolingParameter_PoolMethod_MAX:
		    ContextPoolForwardMAXTOP_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, argmax_data);
		    break;
		  case ContextPoolingParameter_PoolMethod_AVE:
		    ContextPoolForwardAVETOP_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, arg_count);
		}
		else if (this->layer_param_.context_pooling_param().mode() == "bottom")
		{
			switch (this->layer_param_.context_pooling_param().pool())
			{
				case ContextPoolingParameter_PoolMethod_MAX:
				  ContextPoolForwardMAXBOTTOM_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, argmax_data);
				  break;
				case ContextPoolingParameter_PoolMethod_AVE:
				  ContextPoolForwardAVEBOTTOM_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(), bottom_data,channels_, height_, width_, context_height_, context_width_, top_data, arg_count);
		}
	}
}

template <typename Dtype>
void ContextPoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{

	const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  caffe_gpu_set(bottom[0]->count(), Dtype(0.), bottom_diff);
  const int* argmax_data = max_idx_.gpu_data();
  const int* arg_count = arg_count_.gpu_data();
	switch (this->layer_param_.context_pooling_param().mode())
	{
		case ContextPoolingParameter_PoolMode_AROUND:
		switch (this->layer_param_.context_pooling_param().pool())
		{
		  case ContextPoolingParameter_PoolMethod_MAX:
		    ContextPoolBackwardMAXAROUND_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top_diff, argmax_data, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		    break;
		  case ContextPoolingParameter_PoolMethod_AVE:
		    ContextPoolBackwardAVEAROUND_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		    (bottom[0]->count(), top_diff, arg_count, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		}
		break;
		case ContextPoolingParameter_PoolMode_TOP:
		switch (this->layer_param_.context_pooling_param().pool())
		{
		  case ContextPoolingParameter_PoolMethod_MAX:
		    ContextPoolBackwardMAXTOP_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top_diff, argmax_data, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		    break;
		  case ContextPoolingParameter_PoolMethod_AVE:
		    ContextPoolBackwardAVETOP_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		    (bottom[0]->count(), top_diff, arg_count, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		}
		break;
		case ContextPoolingParameter_PoolMode_BOTTOM:
		switch (this->layer_param_.context_pooling_param().pool())
		{
		  case ContextPoolingParameter_PoolMethod_MAX:
		    ContextPoolBackwardMAXBOTTOM_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top_diff, argmax_data, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		    break;
		  case ContextPoolingParameter_PoolMethod_AVE:
		    ContextPoolBackwardAVEBOTTOM_kernel<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		    (bottom[0]->count(), top_diff, arg_count, channels_, height_, width_, context_height_, context_width_, bottom_diff);
		}
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ContextPoolingLayer);
}  // namespace caffe
