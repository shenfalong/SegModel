#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <omp.h>

#include "caffe/data_transformer.hpp"
#include <boost/thread.hpp>

#include <stdint.h>
#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/layers/data/data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void DataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  this->data_transformer_.reset(new DataTransformer<Dtype>(this->transform_param_));

  const int batch_size = this->layer_param_.data_param().batch_size();
  int crop_size = this->layer_param_.transform_param().crop_size();


  db_.reset(new db::DB());
  db_->Open(this->layer_param_.data_param().source(0), db::READ);
  cursor_.reset(db_->NewCursor());

  cv_img.resize(batch_size);
  datum = new Datum[batch_size];
  rand_length.resize(batch_size);
	
	
	
	this->transformed_data_.Reshape(1, 3, crop_size, crop_size);


	this->prefetch_data_.Reshape(batch_size, 3, crop_size, crop_size);
	this->prefetch_label_.Reshape(batch_size,1,1,1);

  sum = 0;
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void DataLayer<Dtype>::InternalThreadEntry(int gpu_id) 
{
	//Please get around from GPU memory in this funtion, as nccl is not thread safe, which will
	//induce a dead block sometimes.
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[gpu_id]));
  

  const int batch_size = this->layer_param_.data_param().batch_size();
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int augment_size = this->layer_param_.transform_param().augment_size();
	const bool random_skip = this->layer_param_.data_param().random_skip();
	const bool random_scale= this->layer_param_.transform_param().random_scale();
	
	
	//for (int i=0;i<this->prefetch_label_.size();i++)
	//	caffe_set(this->prefetch_label_[i]->count(),Dtype(0),this->prefetch_label_[i]->mutable_cpu_data());
  //******************* load from disk **********************
  cv_img.clear();
  for ( int item_id = 0; item_id < batch_size; ++item_id )
  {
    Datum temp_datum;
    temp_datum.ParseFromString(cursor_->value());
    datum[item_id]=temp_datum;

    int rand_skip = random_skip? caffe_rng_rand() % 10+1:1;  
    for(int i_skip=0;i_skip<rand_skip;i_skip++)
    {
      cursor_->Next();
      if (!cursor_->valid())
        cursor_->SeekToFirst();
    }
  }

    
	if (random_scale)
	{
    for(int item_id=0;item_id<batch_size;item_id++)
      rand_length[item_id]	= 256 + caffe_rng_rand() % 224;
	}
	else
	{
    for(int item_id=0;item_id<batch_size;item_id++)
      rand_length[item_id]	= 256 ;
  }
  //********************* image augmentation *****************
  //CPUTimer timer;
  //timer.Start();
  #pragma omp parallel for
  for ( int item_id = 0; item_id < batch_size; ++item_id )
  {
    cv_img[item_id]=DecodeDatumToCVMat(datum[item_id], true);
  }
  

  //timer.Stop();
  //LOG(INFO) << "convert time = " << timer.MilliSeconds() << " ms.";
	//CHECK_GE(500,timer.MilliSeconds());

  //timer.Start();
  //multi-thread seems to have trouble with opencv's functions
#if 1
	if (random_scale)
	{
		#pragma omp parallel for
		for ( int item_id = 0; item_id < batch_size; ++item_id )
		{
		  int width=cv_img[item_id].cols;
		  int height=cv_img[item_id].rows;
		  int short_size = rand_length[item_id];
		  if(width < height)
		  {
		    height = Dtype(short_size) / Dtype(width) * Dtype(height);
		    width = short_size;
		  }
		  else
		  {
		    width = Dtype(short_size) / Dtype(height) * Dtype(width);
		    height = short_size;
		  }
		  cv::resize(cv_img[item_id],cv_img[item_id],cv::Size(width,height),0,0,CV_INTER_LINEAR);
		}
	}	
#endif  
   //timer.Stop();
   //LOG(INFO) << "resize time = " << timer.MilliSeconds() << " ms.";


  //********************* feed into the network *************
  for ( int item_id = 0; item_id < batch_size; ++item_id )
  {
    this->transformed_data_.set_cpu_data(this->prefetch_data_.mutable_cpu_data() + this->prefetch_data_.offset(item_id));
    if (this->layer_param_.transform_param().simple() == false)
    	this->data_transformer_->Transform(cv_img[item_id], &(this->transformed_data_));
    else
    	this->data_transformer_->Transformsimple(cv_img[item_id], &(this->transformed_data_));
    this->prefetch_label_.mutable_cpu_data()[item_id] = datum[item_id].label();
  }
}

template <typename Dtype>
DataLayer<Dtype>::~DataLayer<Dtype>() 
{
  if (this->thread_.get() != NULL && this->thread_->joinable())
    this->thread_->join();
  delete[] datum;
}

INSTANTIATE_CLASS(DataLayer);
REGISTER_LAYER_CLASS(Data);

}  // namespace caffe
