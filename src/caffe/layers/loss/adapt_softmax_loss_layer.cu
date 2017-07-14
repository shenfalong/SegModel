#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/loss/adapt_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

template <typename Dtype>
static int qcompare(const void * a, const void * b)
{
   return *(Dtype*)a - *(Dtype*)b;
}

template <typename Dtype>
static int compare(Dtype a, Dtype b)
{
   return a < b;
}

template <typename Dtype>
static __global__ void sample_kernel(int count, int num, int channels, int height, int width, const Dtype * score, const Dtype * label,const int ignore_label, Dtype * temp_prob, Dtype * num_compute)
{
	CUDA_KERNEL_LOOP(index, count) 
	{
		int n = index / (height * width);
		int h = index / width % height;
		int w = index % width; 
		int c = label[(n * height*4  +  h*4)*width*4  +  w*4];
		if (c == ignore_label)
		{
			temp_prob[index] = 1;	
			num_compute[index] = 0;	
		}	
		else
		{
			temp_prob[index] = score[((n*channels + c)* height*4 + h*4) * width*4 + w*4];			
			num_compute[index] = 1;	
		}
	}	
}

template <typename Dtype>
static __global__ void flag_kernel(int count, int num, int channels, int height, int width, const Dtype * score, const Dtype * label, Dtype threshold, const int ignore_label, Dtype * flag)
{
	CUDA_KERNEL_LOOP(index, count) 
	{
		int n = index / (height * width);
		int h = index / width % height;
		int w = index % width; 
		int c = label[(n * height*4  +  h*4)*width*4  +  w*4];
		if (c == ignore_label)
			flag[index] = 1;
		else
			flag[index] = score[((n * channels + c) * height*4  +  h*4)*width*4  +  w*4] < threshold;
	}	
}

template <typename Dtype>
static __global__ void SoftmaxLossForwardGPU(const int nthreads,
          const Dtype* prob_data, const Dtype* label, Dtype* loss,
          const int num, const int channels, const int height, const int width,
          const bool has_ignore_label_, const int ignore_label_, const Dtype * flag,
          Dtype* counts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int n = index / width / height;
		int h = index / width % height;
		int w = index % width; 
    int label_value  = label[(n * height + h) * width + w];
    if ( (has_ignore_label_ && label_value  == ignore_label_) || (flag[(n * height/4 + h/4) * width/4 + w/4] == 0)) 
    {
      loss[index] = 0;
      counts[index] = 0;
    } 
    else 
    {
    	if (label_value < 0 || label_value >= channels)
    	{
    		printf("label %d not valid.\n",label_value);
    		return;
    	}
    	
      loss[index] = - logf(max(prob_data[( (n * channels + label_value ) * height + h) * width + w], Dtype(FLT_MIN)));
      counts[index] = 1;
    }
  }
}
template <typename Dtype>
static __global__ void SoftmaxLossBackwardGPU(const int nthreads, const Dtype* top,const Dtype *prob_data,
          const Dtype* label, Dtype* bottom_diff, const int num, const int channels,
          const int height, const int width, const bool has_ignore_label_,
          const int ignore_label_,const Dtype * flag, Dtype* counts) 
{
  CUDA_KERNEL_LOOP(index, nthreads) 
  {
    int n = index / width / height;
		int h = index / width % height;
		int w = index % width; 
    int label_value = label[(n * height + h) * width + w]; 
    if ( (has_ignore_label_ && label_value  == ignore_label_) || (flag[(n * height/4 + h/4) * width/4 + w/4] == 0)) 
    {
      counts[index] = 0;
      for (int c = 0; c < channels; ++c)
      	bottom_diff[( (n * channels + c ) * height + h) * width + w]  = 0;     	
    } 
    else 
    {
      bottom_diff[( (n * channels + label_value ) * height + h) * width + w] -= 1;
      counts[index] = 1;
    }
  }
}

template <typename Dtype>
void AdaptSoftmaxWithLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	  
    
  softmax_layer_->Forward_gpu(softmax_bottom_vec_, softmax_top_vec_);


  if (has_ignore_label_) 
  {
		sample_kernel<Dtype><<<CAFFE_GET_BLOCKS(temp_prob.count()),CAFFE_CUDA_NUM_THREADS>>>
		(temp_prob.count(), temp_prob.num(), prob_.channels(), temp_prob.height(), temp_prob.width(),prob_.gpu_data(), bottom[1]->gpu_data(), ignore_label_, 
																																														temp_prob.mutable_gpu_data(),sub_counts_.mutable_gpu_data());	
	}
	else
		LOG(FATAL)<<"what is the igore_label?";	
	 
	Dtype count;
	caffe_gpu_asum(sub_counts_.count(),sub_counts_.gpu_data(),&count);
  std::sort(temp_prob.mutable_cpu_data(),temp_prob.mutable_cpu_data() + temp_prob.count(),compare<Dtype>);
  
  
	int index = floor(count * portion); 
	index = std::min(temp_prob.count()-1,index);
	Dtype threshold = temp_prob.cpu_data()[index];
	
	//LOG(INFO)<<"threshold = "<<threshold<<" index = "<<index;
	
  flag_kernel<Dtype><<<CAFFE_GET_BLOCKS(flag.count()),CAFFE_CUDA_NUM_THREADS>>>
  (flag.count(), flag.num(), prob_.channels(), flag.height(), flag.width(), prob_.gpu_data(),bottom[1]->gpu_data(), threshold, ignore_label_, flag.mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;
  
  //for(int i=0;i<flag.count();i++)
  //{
  //	LOG(INFO)<<" i = "<<i;
  //	CHECK_EQ(flag.cpu_data()[i],1);
  //}
  
  const Dtype* prob_data = prob_.gpu_data();
  const Dtype* label = bottom[1]->gpu_data();
  Dtype* loss_data = loss_.mutable_gpu_data();
  Dtype * count_data = counts_.mutable_gpu_data();
  SoftmaxLossForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[1]->count()),CAFFE_CUDA_NUM_THREADS>>>
  (bottom[1]->count(), prob_data, label, loss_data,
  bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
  has_ignore_label_, ignore_label_,flag.gpu_data(), count_data);
  CUDA_POST_KERNEL_CHECK;
  
  Dtype loss;
  caffe_gpu_asum(bottom[1]->count(), loss_data, &loss);
  
  caffe_gpu_asum(bottom[1]->count(), count_data, &count);
  
  if (count > 0) 
    loss /= count;
  else 
    loss = 0;
  
  
  
  top[0]->mutable_cpu_data()[0] = loss; 
	
}

template <typename Dtype>
void AdaptSoftmaxWithLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
    
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom_diff);
    const Dtype* prob_data = prob_.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->gpu_data();
    Dtype* count_data = counts_.mutable_gpu_data();
    SoftmaxLossBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(bottom[1]->count()),CAFFE_CUDA_NUM_THREADS>>>
    (bottom[1]->count(), top_data,prob_data, label, bottom_diff,
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), 
    has_ignore_label_, ignore_label_,flag.gpu_data(), count_data);
    CUDA_POST_KERNEL_CHECK;
    
    const Dtype loss_weight = top[0]->cpu_diff()[0];
 
    Dtype count;
    caffe_gpu_asum(bottom[1]->count(), count_data, &count);
    if (count > 0) 
      caffe_gpu_scal(prob_.count(), loss_weight / count, bottom_diff);
}

template <typename Dtype>
void AdaptSoftmaxWithLossLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{	  
}

INSTANTIATE_LAYER_GPU_FUNCS(AdaptSoftmaxWithLossLayer);

}  // namespace caffe
