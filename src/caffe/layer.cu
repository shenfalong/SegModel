#include "caffe/layer.hpp"
#include "caffe/solver.hpp"
#include<cfloat>
namespace caffe {
template <typename Dtype>
static __global__ void scale_kernel(int count, int image_dim, Dtype sec_loss_weight, Dtype norm_value, 
																			const Dtype *in, const Dtype *coef, Dtype *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / image_dim;
		out[i] = 2 * sec_loss_weight  *(coef[n]-norm_value)/ coef[n] * in[i];	
	} 
}
template <typename Dtype>
static __global__ void compute_sum(int image_dim, const Dtype *in, Dtype *out)
{
	__shared__ Dtype buffer[CAFFE_CUDA_NUM_THREADS];

	buffer[threadIdx.x] = 0;
	for (int i = threadIdx.x;i < image_dim;i += blockDim.x)
		buffer[threadIdx.x] += in[blockIdx.x*image_dim+i]*in[blockIdx.x*image_dim+i];
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			buffer[threadIdx.x] += buffer[threadIdx.x+s];
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		out[blockIdx.x] = sqrt(buffer[0]);
}
template <typename Dtype>
void Layer<Dtype>::compute_sec_loss(const vector<Blob<Dtype>*>& top, const Dtype sec_loss_weight, const Dtype norm_value)
{
	vector<shared_ptr<Blob<Dtype> > > sum_;
	sum_.resize(top.size());
	for (int i=0;i < top.size();i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i%NGPUS]));
		int num = top[i]->num();
		int channels = top[i]->channels();
		int height = top[i]->height();
		int width = top[i]->width();	
		
		sum_[i].reset(new Blob<Dtype>(num,1,1,1));
		compute_sum<<<num,CAFFE_CUDA_NUM_THREADS>>>
		(channels*height*width,top[i]->gpu_diff(),sum_[i]->mutable_gpu_data());
		
		if (Solver<Dtype>::iter() % 1000 == 0)
		{
			Dtype sum = 0;
			for (int iter = 0;iter<num;iter++)
				sum += sum_[i]->cpu_data()[iter];
			LOG(INFO)<<"sum = "<<sum/Dtype(num);
		}
		scale_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[i]->count(), channels*height*width, sec_loss_weight, norm_value,
		top[i]->gpu_diff(), sum_[i]->gpu_data(), top[i]->mutable_gpu_sec_diff());	
		
		caffe_gpu_scal(top[i]->count(),Dtype(1)/Dtype(num),top[i]->mutable_gpu_sec_diff());
	}
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}
//----------------------------------------- proto <->  memory--------------------
template <typename Dtype>
void Layer<Dtype>::ToProto(LayerParameter* param, bool write_diff) 
{
  param->Clear();
  param->CopyFrom(layer_param_);
  param->clear_blobs();
  for (int i = 0; i < blobs_.size(); ++i) 
    blobs_[i]->ToProto(param->add_blobs(), write_diff);
  
}

template <typename Dtype>
inline Dtype Layer<Dtype>::Forward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();

	Forward_gpu(bottom, top);

	Dtype loss_weight = layer_param_.include().loss_weight();
	
	Dtype loss = 0;
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());
		for (int i=0;i<top.size();i++)
			loss += top[i]->cpu_data()[0] * loss_weight / Dtype(top.size());
	}

	return loss;
};


template <typename Dtype>
inline void Layer<Dtype>::Backward(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	Dtype loss_weight = layer_param_.include().loss_weight();
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());	
		for (int i=0;i<top.size();i++)
			top[i]->mutable_cpu_diff()[0] = loss_weight / Dtype(top.size());
	}
	
  Backward_gpu(top, bottom);	
};

template <typename Dtype>
inline void Layer<Dtype>::SecForward(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();
	Dtype sec_loss_weight = layer_param_.include().sec_loss_weight();
	Dtype norm_value = layer_param_.include().norm_value();
	
	if (sec_loss_weight > 0 && Caffe::second_pass())
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());	
		compute_sec_loss(top,sec_loss_weight / Dtype(top.size()), norm_value);
	}
	else
	{
		SecForward_gpu(bottom, top);
	}
};

INSTANTIATE_CLASS(Layer);
}
