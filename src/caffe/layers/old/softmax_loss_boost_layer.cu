#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_boost_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
struct point
{
	int index;
	Dtype value;
};

template <typename Dtype>
static int compare(point<Dtype> a, point<Dtype> b)
{
   return a.value > b.value;
}

template <typename Dtype>
static __global__ void SoftmaxLossForwardGPUBoost(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int dim, const int spatial_dim,
          const bool has_ignore_label_, const int ignore_label_,
          Dtype* counts) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n%(num/2) * spatial_dim + s]);
    if (has_ignore_label_ && label_value == ignore_label_) 
    {
      loss[index] = 0;
      counts[index] = 0;
    } 
    else 
    {
      loss[index] = -log(max(prob_data[n * dim + label_value * spatial_dim + s], Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
static __global__ void SoftmaxLossBackwardGPUBoost(const int nthreads, const Dtype* top,
          const Dtype* label, Dtype* bottom_diff, const int num, const int dim,
          const int spatial_dim, const bool has_ignore_label_,
          const int ignore_label_, const Dtype * flag, Dtype* counts) 
{
  const int channels = dim / spatial_dim;

  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    const int n = index / spatial_dim;
    const int s = index % spatial_dim;
    const int label_value = static_cast<int>(label[n%(num/2) * spatial_dim + s]);

    if (has_ignore_label_ && label_value == ignore_label_ || !flag[index]) 
    {
      for (int c = 0; c < channels; ++c) 
        bottom_diff[n * dim + c * spatial_dim + s] = 0;
        
      counts[index] = 0;
    } 
    else 
    {
      bottom_diff[n * dim + label_value * spatial_dim + s] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void SoftmaxWithLossBoostLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);

  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();

  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  Dtype* loss_data = loss_.mutable_gpu_data();
  Dtype* count_data = counts_.mutable_gpu_data();
 
  SoftmaxLossForwardGPUBoost<Dtype><<<CAFFE_GET_BLOCKS(num * height * width), CAFFE_CUDA_NUM_THREADS>>>
  (num * height * width, prob_data, label, loss_data, num, channels*height*width, height*width, has_ignore_label_, ignore_label_, count_data);
  
  CHECK_EQ(num%4,0);
  CHECK_EQ(height,1);
  CHECK_EQ(width,1);
	std::vector<point<Dtype> > sort_loss;
	sort_loss.resize(num/2);
	for (int i=0;i<num/2;i++)
	{
		sort_loss[i].index = i;
		sort_loss[i].value = loss_.cpu_data()[num/2 + i];
	}
	std::sort(sort_loss.begin(),sort_loss.end(),compare<Dtype>);
  
  caffe_set(flag.count(),Dtype(0),flag.mutable_cpu_data());
  for (int i=0;i<num/4*1.5;i++)
		flag.mutable_cpu_data()[sort_loss[i].index]=1;
  
  
  
  
  Dtype loss = 0;
  Dtype count = 0;
  for (int i=0;i<num/2;i++)
  {
  	if (flag.cpu_data()[i] == 1)
  	{
  		loss += loss_.cpu_data()[i];
  		count ++;
  	}	
  }
  
  
 
  
  //LOG(INFO)<<" num = "<<num<<" height = "<<height<<" width = "<<width;
  
	if (count > 0)
  	top[0]->mutable_cpu_data()[0] = loss / count;
  else
  	top[0]->mutable_cpu_data()[0] = 0; 	 
 		
}

template <typename Dtype>
void SoftmaxWithLossBoostLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{

  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* top_data = top[0]->gpu_data();
  caffe_copy(prob_.count(), prob_data, bottom_diff);
  const Dtype* label = bottom[1]->gpu_data();
	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  Dtype* count_data = counts_.mutable_gpu_data();

  SoftmaxLossBackwardGPUBoost<Dtype><<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
  (num*height*width, top_data, label, bottom_diff, num, channels*height*width, height*width, has_ignore_label_, ignore_label_,flag.gpu_data(), count_data);

  Dtype count;
  caffe_gpu_asum(num * height * width, count_data, &count);
  
  const Dtype loss_weight = top[0]->cpu_diff()[0] / count;
  caffe_gpu_scal(prob_.count(), loss_weight , bottom_diff);

}

INSTANTIATE_LAYER_GPU_FUNCS(SoftmaxWithLossBoostLayer);

}  // namespace caffe
