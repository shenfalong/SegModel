#include <vector>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/layers/operator/fast_guided_crf_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
using std::max;
using std::min;
namespace caffe 
{
template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
  maxIter = this->layer_param_.crf_param().max_iter();
  radius=this->layer_param_.crf_param().radius();
  alpha=this->layer_param_.crf_param().alpha();
  eps=this->layer_param_.crf_param().eps();
  nodeBel.resize(maxIter);


  for(int iter=0;iter<maxIter;iter++)
    nodeBel[iter]=new Blob<Dtype>();

  if (this->blobs_.size() > 0)
    LOG(INFO)<<"skip initialization";
  else
  {
    int channels = bottom[0]->channels();
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(1,1,channels,channels));
    caffe_set(this->blobs_[0]->count(),Dtype(1),this->blobs_[0]->mutable_cpu_data());
    for(int c=0;c<channels;c++)
      this->blobs_[0]->mutable_cpu_data()[c*channels+c]=0;
  }
}

template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

  top[0]->ReshapeLike(*bottom[0]);
  int num = bottom[0]->num();
  int maxStates = bottom[0]->channels();
  int channels = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	if (maxStates != this->blobs_[0]->width())
	{
		LOG(INFO)<<"channels not match "<<maxStates<<" vs "<<this->blobs_[0]->width();
		this->blobs_[0].reset(new Blob<Dtype>(1,1,maxStates,maxStates));
			
		caffe_set(this->blobs_[0]->count(),Dtype(1),this->blobs_[0]->mutable_cpu_data());
    for(int c=0;c<maxStates;c++)
      this->blobs_[0]->mutable_cpu_data()[c*maxStates+c]=0;
	}

  filterPot.Reshape(num,maxStates,height,width);
  compatPot.Reshape(num,maxStates,height,width);
  tempPot.Reshape(num,maxStates,height,width);
  output_p1.Reshape(num,maxStates,height,width);
  output_p2.Reshape(num,maxStates,height,width);
  for(int iter=0;iter<maxIter;iter++)
    nodeBel[iter]->Reshape(num,maxStates,height,width);

	
	mean_a_up.Reshape(num*maxStates,channels,height,width);
  mean_b_up.Reshape(num*maxStates,1,height,width);
	
	I_sub.Reshape(num,3,height/4,width/4);
	temp_sub.Reshape(num*maxStates,1,height/4,width/4);
	
  mean_I.Reshape(num,channels,height/4,width/4);
  II.Reshape(num,channels*channels,height/4,width/4);
  mean_II.Reshape(num,channels*channels,height/4,width/4);
  var_I.Reshape(num,channels*channels,height/4,width/4);
  inv_var_I.Reshape(num,channels*channels,height/4,width/4);
  mean_p.Reshape(num*maxStates,1,height/4,width/4);
  Ip.Reshape(num*maxStates,channels,height/4,width/4);
  mean_Ip.Reshape(num*maxStates,channels,height/4,width/4);
  cov_Ip.Reshape(num*maxStates,channels,height/4,width/4);
  a.Reshape(num*maxStates,channels,height/4,width/4);
  b.Reshape(num*maxStates,1,height/4,width/4);
  mean_a.Reshape(num*maxStates,channels,height/4,width/4);
  mean_b.Reshape(num*maxStates,1,height/4,width/4);
  buffer_image.Reshape(num,channels,height/4,width/4);
  buffer_score.Reshape(num,maxStates,height/4,width/4);
  buffer_image_score.Reshape(num,channels*maxStates,height/4,width/4);
  buffer_image_image.Reshape(num,channels*channels,height/4,width/4);
  
  
}



template <typename Dtype>
FastGuidedCRFLayer<Dtype>::~FastGuidedCRFLayer()
{
	for(int iter=0;iter<maxIter;iter++)
    delete nodeBel[iter];
  nodeBel.clear();
}

 

INSTANTIATE_CLASS(FastGuidedCRFLayer);
REGISTER_LAYER_CLASS(FastGuidedCRF);
}  // namespace caffe
