#include <vector>
#include <float.h>
#include "caffe/layers/operator/cudnn_batch_norm_layer.hpp"
#define BN_EPS Dtype(CUDNN_BN_MIN_EPSILON)


namespace caffe {
//---------------------------------- forward ---------------
template <typename Dtype>
static __global__ void kernel_local_stats(int num, int channels, int spatial_dim, const Dtype norm_factor, const Dtype* bottom_data, 
																					Dtype* mean, Dtype* var) 
{
  // store local E[x] to mean, E[x^2] to var temporarily
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_data[index] / norm_factor;
    buffer2[tid] += bottom_data[index] * bottom_data[index] / norm_factor;
  }
  __syncthreads();
  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    mean[c] = buffer1[0] ;
    var[c] = buffer2[0] ;
  }
}
template <typename Dtype>
static __global__ void kernel_forward( const int count, const int num, const int channels, const int spatial_dim, 
		const Dtype* mean, const Dtype* var,  const Dtype* bottom_data,  Dtype* top_data) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
    int c = (index / spatial_dim) % channels;
    top_data[index] = (bottom_data[index] - mean[c]) / sqrt(var[c] + BN_EPS);
  }
}
//------------------------ backward -------
template <typename Dtype>
static __global__ void kernel_backward_mean_var(const int num, const int channels, const int spatial_dim,
		const Dtype* top_diff, const Dtype* bottom_data, const Dtype * mean_data, const Dtype * var_data, Dtype* mean_diff, Dtype* var_diff) 
{
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += top_diff[index] / sqrt(var_data[c] + BN_EPS);
    buffer2[tid] += top_diff[index] * (bottom_data[index] - mean_data[c]) / sqrt(var_data[c] + BN_EPS);   
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    mean_diff[c] = - buffer1[0];
    var_diff[c] =  - buffer2[0] / (2*(var_data[c] + BN_EPS));
  }
}
template <typename Dtype>
static __global__ void kernel_backward_bottom_0(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, const Dtype* top_diff, 
			const Dtype* var, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const Dtype inv_std = Dtype(1) / sqrt(var[c] + BN_EPS);
    bottom_diff[index]  =  inv_std * top_diff[index];    							
  }
}
template <typename Dtype>
static __global__ void kernel_backward_bottom_1(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, const Dtype* top_diff, 
			const Dtype* var, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const Dtype inv_std = Dtype(1) / sqrt(var[c] + BN_EPS);
    bottom_diff[index]  +=  inv_std * top_diff[index];    							
  }
}
template <typename Dtype>
static __global__ void kernel_mean_var_backward_bottom(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, 
		const Dtype * mean_data, const Dtype* var_data,const Dtype * mean_diff, const Dtype * var_diff, 
		const Dtype* bottom_data, Dtype* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    bottom_diff[index] += mean_diff[c] / norm_factor 
    									 + var_diff[c] / norm_factor * Dtype(2) * (bottom_data[index] - mean_data[c]);
  }
}
//----------------------------------------secforward-----------------------------
//------------------------ diff ------------------
template <typename Dtype>
static __global__ void kernel_secforward_diff_mean_diff_var(const int num, const int channels, const int spatial_dim, const int norm_factor,
		const Dtype* bottom_sec_diff, const Dtype* bottom_data, const Dtype * mean_data, const Dtype * var_data, Dtype* mean_sec_diff, Dtype* var_sec_diff) 
{
  __shared__ Dtype buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_sec_diff[index];
    buffer2[tid] += bottom_sec_diff[index] * (bottom_data[index] - mean_data[c]);   
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    mean_sec_diff[c] =  buffer1[0]  / norm_factor;
    var_sec_diff[c] =   buffer2[0] * Dtype(2) / norm_factor;
  }
}
template <typename Dtype>
static __global__ void kernel_secforward_top(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, const Dtype* bottom_sec_diff, 
			const Dtype* var, Dtype* top_sec_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const Dtype inv_std = Dtype(1) / sqrt(var[c] + BN_EPS);
    top_sec_diff[index]  =  inv_std * bottom_sec_diff[index];    							
  }
}
template <typename Dtype>
static __global__ void kernel_diff_mean_diff_var_secforward_top(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, 
		const Dtype * mean_data, const Dtype* var_data,const Dtype * mean_sec_diff, const Dtype * var_sec_diff, 
		const Dtype* bottom_data, Dtype* top_sec_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    top_sec_diff[index] += - mean_sec_diff[c] / sqrt(var_data[c]+BN_EPS)
    									 - var_sec_diff[c] * (bottom_data[index] - mean_data[c]) / pow(var_data[c]+BN_EPS, Dtype(1.5)) * Dtype(0.5);
  }
}

//------------------------- data --------------
template <typename Dtype>
static __global__ void kernel_secforward_bottom(const int num, const int channels, const int spatial_dim, const Dtype norm_factor, 
  const Dtype * bottom_sec_diff, const Dtype * top_diff,
  const Dtype * var_data, const Dtype * var_diff, const Dtype * var_sec_diff,
  Dtype * bottom_diff)
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    bottom_diff[index] = bottom_sec_diff[index]*var_diff[c]*Dtype(2)/norm_factor
    									 - var_sec_diff[c]*top_diff[index]/pow(var_data[c]+BN_EPS,Dtype(1.5))*Dtype(0.5); 		 
  }
}
template <typename Dtype>
static __global__ void kernel_secforward_mean_var(const int num, const int channels, const int spatial_dim, const Dtype norm_factor,
		const Dtype * bottom_sec_diff, const Dtype * top_diff, const Dtype * bottom_data,
		const Dtype * mean_data, const Dtype * mean_sec_diff, const Dtype * var_data,	 const Dtype * var_sec_diff,
		Dtype * mean_diff, Dtype * var_diff)
{
	__shared__ Dtype buffer_secx[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer_dy[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer_secx_dy[CAFFE_CUDA_NUM_THREADS];
  __shared__ Dtype buffer_x_dy[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer_secx[tid] = buffer_dy[tid] = buffer_secx_dy[tid] = buffer_x_dy[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer_secx[tid] += bottom_sec_diff[index];
    buffer_dy[tid] += top_diff[index]; 
    buffer_secx_dy[tid] += bottom_sec_diff[index]*top_diff[index];
    buffer_x_dy[tid] += (bottom_data[index] - mean_data[c])*top_diff[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer_secx[tid] += buffer_secx[tid + s];
      buffer_dy[tid] += buffer_dy[tid + s];
      buffer_secx_dy[tid] += buffer_secx_dy[tid + s];
      buffer_x_dy[tid] += buffer_x_dy[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    mean_diff[c] = -buffer_secx[0]*var_diff[c]*Dtype(2)/norm_factor+var_sec_diff[c]*buffer_dy[0]/pow(var_data[c]+BN_EPS,Dtype(1.5))*Dtype(0.5); 
    var_diff[c] = -buffer_secx_dy[0]/pow(var_data[c]+BN_EPS,Dtype(1.5))*Dtype(0.5)
    							+mean_sec_diff[c]*buffer_dy[0]/pow(var_data[c]+BN_EPS,Dtype(1.5))*Dtype(0.5)
    							+var_sec_diff[c]*buffer_x_dy[0]/pow(var_data[c]+BN_EPS,Dtype(2.5))*Dtype(0.75); 		 
  }
}	
//----------------------------------------------------
//----------------------------------------------------------
template <typename Dtype>
static __global__ void linear_batch_norm_forward(int num,int channels,int height,int width,
													const Dtype *weight,const Dtype * in, const Dtype * bias, Dtype *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind] + bias[c];
  }
}

template <typename Dtype>
static __global__ void linear_batch_norm_backward(int num,int channels,int height,int width,
													const Dtype *weight,const Dtype * in, const Dtype * bias, Dtype *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind];
  }
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	Blob<Dtype> temp;
	temp.ReshapeLike(*bottom[0]);
#if 1
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();


	kernel_local_stats<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,
		static_cast<Dtype>(num * height * width),
		bottom[0]->gpu_data(),
		savedmean.mutable_gpu_data(), savedinvvariance.mutable_gpu_data());

	caffe_gpu_mul(channels, savedmean.gpu_data(), savedmean.gpu_data(), savedmean.mutable_gpu_sec_diff()); 
	caffe_gpu_sub(channels, savedinvvariance.gpu_data(), savedmean.gpu_sec_diff(), savedinvvariance.mutable_gpu_data());
 	
 	
	kernel_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( bottom[0]->count(), num, channels, height * width,
		savedmean.gpu_data(), savedinvvariance.gpu_data(),
		bottom[0]->gpu_data(),
		temp.mutable_gpu_data());	
	

	if (Caffe::bn_state() == FROZEN)
	{
		const int K = bottom[0]->channels();
		weights.Reshape(1,K,1,1);
		bias.Reshape(1,K,1,1);
	
		for(int c=0;c<K;c++)
		{
			weights.mutable_cpu_data()[c] = this->blobs_[0]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c]+ Dtype(CUDNN_BN_MIN_EPSILON)));
			bias.mutable_cpu_data()[c] = -this->blobs_[0]->cpu_data()[c]*this->blobs_[2]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c] + Dtype(CUDNN_BN_MIN_EPSILON)))
																								+this->blobs_[1]->cpu_data()[c];															
		}				
	} 	

	if (Caffe::number_collect_sample == 0 && Caffe::bn_state() == LEARNED)
	{
		caffe_gpu_set(this->blobs_[2]->count(),Dtype(0),this->blobs_[2]->mutable_gpu_data());
		caffe_gpu_set(this->blobs_[3]->count(),Dtype(0),this->blobs_[3]->mutable_gpu_data());
	}

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  
   
	if (Caffe::bn_state() == LEARNED)
	{	
		double factor;
		if (Caffe::number_collect_sample == -1)
			factor = 0.01;
		else 
			factor = double(1)/double(Caffe::number_collect_sample+1);
	

		CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      factor,
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(CUDNN_BN_MIN_EPSILON),
		      NULL,NULL));	       
  }  
	else
	{

		linear_batch_norm_forward<Dtype><<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),bottom[0]->gpu_data(),bias.gpu_data(),top[0]->mutable_gpu_data()); 
/*
		CUDNN_CHECK(cudnnBatchNormalizationForwardInference(Caffe::parallel_cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(0.001)
		      ));	       	       
*/	         
	}	   	    
	for (int i=0;i<temp.count();i++)
	{
		if (std::abs(temp.cpu_data()[i]-top[0]->cpu_data()[i])>1e-1)
		{
			int c = (i / (height*width)) % channels;
			LOG(ERROR)<<bottom[0]->cpu_data()[i]<<", "<<savedmean.cpu_data()[c]<<", "<<savedinvvariance.cpu_data()[c];
			LOG(FATAL)<<temp.cpu_data()[i]<<", "<<top[0]->cpu_data()[i];
		}
	}
#endif           
}

template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
#if 1
	if (Caffe::bn_state() == LEARNED)
  {
 
		const Dtype* top_data = top[0]->gpu_data();
		const Dtype* top_diff = top[0]->gpu_diff();
		const Dtype* bottom_data = bottom[0]->gpu_data();
		Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (Caffe::frozen_param() == false)
		{
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),
						double(CUDNN_BN_MIN_EPSILON),
						NULL,NULL));		
		}
		else
		{
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_sec_diff(),this->blobs_[1]->mutable_gpu_sec_diff(),//not use
						double(CUDNN_BN_MIN_EPSILON),
						NULL,NULL));		
		}

  }    
  else
  {
  	linear_batch_norm_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),top[0]->gpu_diff(),bias.gpu_data(),bottom[0]->mutable_gpu_diff());  
  } 
#else
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  kernel_local_stats<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width, static_cast<Dtype>(num * height * width),
		bottom[0]->gpu_data(),
		savedmean.mutable_gpu_data(),
		savedinvvariance.mutable_gpu_data());
  
  caffe_gpu_mul(channels, savedmean.gpu_data(), savedmean.gpu_data(), savedmean.mutable_gpu_sec_diff()); 
	caffe_gpu_sub(channels, savedinvvariance.gpu_data(), savedmean.gpu_sec_diff(), savedinvvariance.mutable_gpu_data());
  
  kernel_backward_bottom_0<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(num, channels, height * width,  static_cast<Dtype>(num * height * width), top[0]->gpu_diff(),savedinvvariance.gpu_data(),
		bottom[0]->mutable_gpu_diff());  
  
	kernel_backward_mean_var<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,
		top[0]->gpu_diff(), bottom[0]->gpu_data(),savedmean.gpu_data(),savedinvvariance.gpu_data(),
		savedmean.mutable_gpu_diff(), savedinvvariance.mutable_gpu_diff());  
  

	kernel_mean_var_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,  static_cast<Dtype>(num * height * width),
		savedmean.gpu_data(), savedinvvariance.gpu_data(),savedmean.gpu_diff(), savedinvvariance.gpu_diff(),
		bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());       
#endif
}
template <typename Dtype>
void CuDNNBatchNormLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
//-------------------------------------- diff---------------------------------------
#if 0
	kernel_local_stats<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,
		static_cast<Dtype>(num * height * width),
		bottom[0]->gpu_data(),
		savedmean.mutable_gpu_data(),
		savedinvvariance.mutable_gpu_data());

	caffe_gpu_mul(channels, savedmean.gpu_data(), savedmean.gpu_data(), savedmean.mutable_gpu_sec_diff()); 
	caffe_gpu_sub(channels, savedinvvariance.gpu_data(), savedmean.gpu_sec_diff(), savedinvvariance.mutable_gpu_data());

	kernel_secforward_diff_mean_diff_var<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width, static_cast<Dtype>(num * height * width),
		bottom[0]->gpu_sec_diff(), bottom[0]->gpu_data(),savedmean.gpu_data(),savedinvvariance.gpu_data(),
		savedmean.mutable_gpu_sec_diff(), savedinvvariance.mutable_gpu_sec_diff());

	kernel_secforward_top<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
  (num, channels, height * width,  static_cast<Dtype>(num * height * width), bottom[0]->gpu_sec_diff(),savedinvvariance.gpu_data(),
  top[0]->mutable_gpu_sec_diff());       	
		
	kernel_diff_mean_diff_var_secforward_top<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,  static_cast<Dtype>(num * height * width),
		savedmean.gpu_data(), savedinvvariance.gpu_data(), savedmean.gpu_sec_diff(), savedinvvariance.gpu_sec_diff(),
		bottom[0]->gpu_data(), top[0]->mutable_gpu_sec_diff());
#else
	CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::zero,
						cudnn::dataType<Dtype>::one,cudnn::dataType<Dtype>::one,
						bottom_desc_, bottom[0]->gpu_data(),
						bottom_desc_,bottom[0]->gpu_sec_diff(),
						top_desc_, top[0]->mutable_gpu_sec_diff(),
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_sec_diff(),//blobs_[1]->diff shoud be fixed
						double(CUDNN_BN_MIN_EPSILON),
						NULL,NULL));
#endif
}
INSTANTIATE_LAYER_GPU_FUNCS(CuDNNBatchNormLayer);

}  // namespace caffe
