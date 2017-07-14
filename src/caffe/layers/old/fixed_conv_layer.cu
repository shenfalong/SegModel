
#include <vector>

#include "caffe/layers/fixed_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <sm_20_atomic_functions.h>
#define TILE_DIM 16

namespace caffe {
/*
template <typename Dtype>
__global__ void shen_conv_forward(int num_channels, int width, int height, Dtype *in, const Dtype* __restrict__ filter, Dtype *out) 
{
	__shared__ float buffer[TILE_DIM + Mask_width - 1][TILE_DIM + Mask_width - 1];
	for (int k = 0; k < num_channels; k++) 
	{
		// First batch loading
		int dest = threadIdx.y * TILE_WIDTH + threadIdx.x;
		int destY = dest / w, destX = dest % w;
		int srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		int srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		int src = (srcY * width + srcX) * channels + k;
		if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
			 buffer[destY][destX] = in[src];
		else
			 buffer[destY][destX] = 0;

		// Second batch loading
		dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
		destY = dest / w, destX = dest % w;
		srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
		srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
		src = (srcY * width + srcX) * channels + k;
		if (destY < w) {
			 if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
					buffer[destY][destX] = in[src];
			 else
					buffer[destY][destX] = 0;
		}
		__syncthreads();

		Dtype sum = 0;
		for (int y = 0; y < Mask_width; y++)
			 for (int x = 0; x < Mask_width; x++)
					sum += buffer[threadIdx.y + y][threadIdx.x + x] * filter[y * Mask_width + x];
					
		int out_y = blockIdx.y * TILE_WIDTH + threadIdx.y;
		int out_x = blockIdx.x * TILE_WIDTH + threadIdx.x;
		if (out_y < height && out_x < width)
			 out[(k * height + out_y) * width + out_x] = sum;
		__syncthreads();
	}
}
*/
																																								
template <typename Dtype>
static __global__ void fixed_conv_forward(const int count, const int num, const int channels, const int height, const int width, 
																					const Dtype *in, const Dtype * __restrict__ filter, const Dtype * __restrict__ weight, Dtype *out)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	int nw = i / width;
  	int nh = nw / height;
  	int nc = nh / channels;

  	int w = i - nw * width;
  	int h = nw - nh * height;
  	int c = nh - nc * channels;
  
  	Dtype sum = 0;
//---------------------------  
#pragma unroll
		for (int k=0;k<9;k++)
		{	
			if (filter[(k*channels+c)*3+0] == 1)
			{
				int y = filter[(k*channels+c)*3+1];
				int x = filter[(k*channels+c)*3+2];
				if (h+y>=0 && h+y<height && w+x>=0 && w+x<width)
					sum += in[i+y*width+x] * weight[k*channels+c];
			}
		}
//----------------------------		
  	out[i] = sum; 	
  }
}
template <typename Dtype>
static __global__ void fixed_conv_backward_data(const int count, const int num, const int channels, const int height, const int width, 
																																const Dtype *diff_out, const Dtype * __restrict__ filter, const Dtype *__restrict__ weight, Dtype *diff_in)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	int nw = i / width;
  	int nh = nw / height;
  	int nc = nh / channels;
  	
  	int w = i - nw * width;
  	int h = nw - nh * height;
  	int c = nh - nc * channels;

  	Dtype sum = 0;
//----------------  	
#pragma unroll
		for (int k=0;k<9;k++)
		{
			if (filter[(k*channels+c)*3+0] == 1)
			{
				int y = filter[(k*channels+c)*3+1];
				int x = filter[(k*channels+c)*3+2];
				if (h-y>=0 && h-y<height && w-x>=0 && w-x<width)
					sum += diff_out[i-y*width-x] * weight[k*channels+c];
			}
		}
//------------------------------------
  	diff_in[i] = sum; 	
  }
}
template <typename Dtype>
static __global__ void fixed_conv_backward_weight(const int count, const int num, const int channels, const int height, const int width, 
																																const Dtype *diff_out, const Dtype * in, const Dtype * __restrict__ filter, Dtype * diff_weight)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	int nw = i / width;
  	int nh = nw / height;
  	int nc = nh / channels;
  	
  	int w = i - nw * width;
  	int h = nw - nh * height;
  	int c = nh - nc * channels;
//----------------  	
#pragma unroll
		for (int k=0;k<9;k++)
		{
			Dtype sum = 0;
			if (filter[(k*channels+c)*3+0] == 1)
			{
				int y = filter[(k*channels+c)*3+1];
				int x = filter[(k*channels+c)*3+2];
				if (h-y>=0 && h-y<height && w-x>=0 && w-x<width)
				{
					for (int n=0;n<num;n++)
						sum += diff_out[n*count+i-y*width-x] * in[n*count+i];
				}
			}
			diff_weight[k*count+i] = sum;
		}
//------------------------------------	
  }
}

template <typename Dtype>
void FixedConvLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  //shen_conv_forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  //(num*channels, width, height, bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), top[0]->mutable_gpu_data()) 
  
	fixed_conv_forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),num,channels,height,width, bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(),top[0]->mutable_gpu_data()); 
}

template <typename Dtype>
void FixedConvLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
//---------------------------------  
  diff_weight_buffer_->Reshape(num*height*width,1,1,channels*9);
  all_one_->Reshape(1, 1, height, width);
  caffe_gpu_set(all_one_->count(),Dtype(1.0),all_one_->mutable_gpu_data());
//---------------------------------    

	fixed_conv_backward_data<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),num,channels,height,width, top[0]->gpu_diff(), 
  																							this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(),bottom[0]->mutable_gpu_diff());  
  
  fixed_conv_backward_weight<Dtype><<<CAFFE_GET_BLOCKS(channels*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (channels*height*width,num,channels,height,width, top[0]->gpu_diff(), bottom[0]->gpu_data(), 
  																							this->blobs_[0]->gpu_data(), diff_weight_buffer_->mutable_gpu_diff()); 

	caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, 9*channels,1, height*width, 
											(Dtype)1., diff_weight_buffer_->gpu_diff(), all_one_->gpu_data(),
											(Dtype)1., this->blobs_[1]->mutable_gpu_diff());																																											
}

INSTANTIATE_LAYER_GPU_FUNCS(FixedConvLayer);
}  // namespace caffe
