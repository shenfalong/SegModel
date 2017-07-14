
#include <vector>
#include "caffe/solver.hpp"
#include "caffe/layers/data/noise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void NoiseLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
		
	caffe_rng_uniform<Dtype>(top[0]->count(), Dtype(-1), Dtype(1), top[0]->mutable_cpu_data());
	if (top.size() > 1)
	{
		for (int i=0;i<top[1]->count();i++)
			top[1]->mutable_cpu_data()[i]=abs(caffe_rng_rand()%classes_);
	}
	
	
  if (bottom.size() == 1)
  {  														
		int channels = top[0]->channels();  								
		for (int n=0;n<top[0]->num();n++)
		{
			for (int c=0;c<top[0]->channels();c++)
			{
				if (c>channels_)
					top[0]->mutable_cpu_data()[n*channels+channels_+c] = Dtype(0);
			} 
			int label = bottom[0]->cpu_data()[n];
			CHECK_LE(label,channels-channels_-1);
			top[0]->mutable_cpu_data()[n*channels+channels_+label] = Dtype(1);
		}								
	}					
}

template <typename Dtype>
void NoiseLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	
}
template <typename Dtype>
void NoiseLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(NoiseLayer);
}  // namespace caffe
