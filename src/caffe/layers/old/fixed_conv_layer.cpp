#include <vector>
#include <set>
#include "caffe/layers/fixed_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FixedConvLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	
	kernel_size_ = this->layer_param_.convolution_param().kernel_size();
	filter_stride_ = this->layer_param_.convolution_param().filter_stride();
	//CHECK_EQ(kernel_size,3);
	
	if (this->blobs_.size() > 0)
    LOG(INFO)<<"skip initialization";
  else
  {
    int channels = bottom[0]->channels();
    this->blobs_.resize(2);
    this->blobs_[0].reset(new Blob<Dtype>(9,channels,1,3));
    this->blobs_[1].reset(new Blob<Dtype>(9,channels,1,1));
 		if (this->layer_param_.convolution_param().random_field())
 		{
			for (int c=0;c<channels;c++)
			{
				set<pair<int,int> > yx_set;
				int y, x;

				yx_set.clear();
				for (int k=0;k<9;k++)
				{
					this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+0] = 1;    	
					do
					{     	 
						y =  caffe_rng_rand()%kernel_size_ - kernel_size_/2; 
						x =  caffe_rng_rand()%kernel_size_ - kernel_size_/2;
				
					} while(yx_set.find(pair<int,int>(y,x)) != yx_set.end());
			
					yx_set.insert(pair<int,int>(y,x));
			
					 
					this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+1] = y;
					this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+2] = x;
				}
			} 		
 		}
 		else
 		{
	 		for (int c=0;c<channels;c++)
	 			for (int k=0;k<9;k++)
			 	{
			 		this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+0] = 1;
			 		this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+1] = (k/3-1)*filter_stride_;
			 		this->blobs_[0]->mutable_cpu_data()[(k*channels+c)*3+2] = (k%3-1)*filter_stride_;
			 	}
 		}   
    caffe_rng_gaussian(this->blobs_[1]->count(), Dtype(0), Dtype(0.02), this->blobs_[1]->mutable_cpu_data()); 
  }
}

template <typename Dtype>
void FixedConvLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
//----------------------------- work space ------------------------- 
	diff_weight_buffer_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_]);
	all_one_ = static_cast<Blob<Dtype> *>(Caffe::parallel_workspace_[gpu_id_+Caffe::GPUs.size()]);
//--------------------------------	
	top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void FixedConvLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void FixedConvLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(FixedConvLayer);
#endif

INSTANTIATE_CLASS(FixedConvLayer);
REGISTER_LAYER_CLASS(FixedConv);
}  // namespace caffe
