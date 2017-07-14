#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

SyncedMemory::~SyncedMemory() 
{
  if (cpu_ptr_ != NULL)
  { 
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  	cpu_ptr_ = NULL;
  }  

  if (gpu_ptr_ != NULL) 
  {
    int initial_device;
    cudaGetDevice(&initial_device);
    if (gpu_device_ != -1) 
      CUDA_CHECK(cudaSetDevice(gpu_device_));
    
    CUDA_CHECK(cudaFree(gpu_ptr_));
    gpu_ptr_ = NULL;
    cudaSetDevice(initial_device);
  }
}

inline void SyncedMemory::to_cpu() 
{
  switch (head_) 
  {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    break;
  case HEAD_AT_GPU:
    if (cpu_ptr_ == NULL) 
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);

    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() 
{
  switch (head_) 
  {
  case UNINITIALIZED:
    CUDA_CHECK(cudaGetDevice(&gpu_device_));
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) 
    {
      CUDA_CHECK(cudaGetDevice(&gpu_device_));
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
}

const void* SyncedMemory::cpu_data() 
{
  to_cpu();
  return (const void*)cpu_ptr_;
}


const void* SyncedMemory::gpu_data() 
{
	to_gpu();
  return (const void*)gpu_ptr_;
}


void* SyncedMemory::mutable_cpu_data() 
{
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() 
{
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) 
{
  if (data != NULL)
  {
		cpu_ptr_ = data;
		head_ = HEAD_AT_CPU;
	}
	else
	{
		cpu_ptr_ = data;
		head_ = UNINITIALIZED;
	}	
}




}  // namespace caffe

