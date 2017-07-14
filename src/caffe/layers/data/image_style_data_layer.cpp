#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/data/image_style_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
void ImageStyleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  TransformationParameter transform_param = this->layer_param_.transform_param();
  DataParameter image_data_param    = this->layer_param_.data_param();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = transform_param.crop_size();

{
  // Read the file with filenames and labels
  const string& source = image_data_param.source(0);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string linestr;
  lines_.clear();
  while (std::getline(infile, linestr))
  {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = imgfn;
    imgfn = "JPEGImages/" + imgfn + ".jpg";
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (image_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines_.begin(), lines_.end());
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
}	
{	
	const string& source = image_data_param.source(1);
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string linestr;
  lines_style_.clear();
  while (std::getline(infile, linestr))
  {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = imgfn;
    imgfn = imgfn + ".jpg";
    lines_style_.push_back(std::make_pair(imgfn, segfn));
  }

  if (image_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines_style_.begin(), lines_style_.end());
  }
  LOG(INFO) << "A total of " << lines_style_.size() << " images.";
  lines_style_id_ = 0;
}	
	
	


  this->prefetch_data_.Reshape(batch_size, 3, crop_size, crop_size);		  		  
  this->prefetch_label_.Reshape(batch_size, 3, crop_size, crop_size);		 

}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageStyleDataLayer<Dtype>::InternalThreadEntry(int gpu_id)
{
 //new thread treat GPU 0 as default device, so it is necessary to set device in case of
  //ghost memory
#ifndef CPU_ONLY
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[gpu_id]));
#endif
	
 	TransformationParameter transform_param = this->layer_param_.transform_param();
  DataParameter image_data_param    = this->layer_param_.data_param();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = transform_param.crop_size();
  string root_folder   = image_data_param.root_folder();
	
	std::vector<float> mean_values_;
	mean_values_.clear();
	if (transform_param.mean_value_size() > 0) 
  {  
    for (int c = 0; c < transform_param.mean_value_size(); ++c) 
      mean_values_.push_back(transform_param.mean_value(c));
  }
	
  const int lines_size = lines_.size();
	const int lines_style_size = lines_style_.size();
	
  for(int item_id = 0; item_id < batch_size; item_id++)
  {
		CHECK_GT(lines_size, lines_id_);
		cv::Mat cv_img;
		cv_img = cv::imread(root_folder + lines_[lines_id_].first, CV_LOAD_IMAGE_COLOR);
		if (!cv_img.data) {LOG(FATAL) << "Fail to load img: " << root_folder + lines_[lines_id_].first;}
		int img_channels = cv_img.channels();
		
		cv::Mat cv_label;
		cv_label = cv::imread("/home/i-shenfalong/datasets/Style/train/" + lines_style_[lines_style_id_].first, CV_LOAD_IMAGE_COLOR);
		//cv_label = cv::imread("/home/i-shenfalong/caffe/newStyle/resnet/styleDataset/JPEGImages/" +format_int(lines_style_id_%11+1)+ ".jpg", CV_LOAD_IMAGE_COLOR);
		if (!cv_label.data) 
		{
			LOG(INFO) << "Fail to load img: " << lines_style_[lines_style_id_].first;
			lines_style_id_++;
			continue;
		}	
{
		int rand_length	= crop_size + 1 + caffe_rng_rand() % 224;
		int width=cv_img.cols;
	  int height=cv_img.rows;
	  if(width < height)
	  {
	    height = Dtype(rand_length) / Dtype(width) * Dtype(height);
	    width = rand_length;
	  }
	  else
	  {
	    width = Dtype(rand_length) / Dtype(height) * Dtype(width);
	    height = rand_length;
	  }
	  cv::resize(cv_img,cv_img,cv::Size(width,height),0,0,CV_INTER_CUBIC);
	  int w_off = caffe_rng_rand()%(width-crop_size);
	  int h_off = caffe_rng_rand()%(height-crop_size);
		cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_img = cv_img(roi);
}    

		
		if (Solver<Dtype>::iter() > 80000)
		{      
			int rand_length	= crop_size + 1 + caffe_rng_rand() % 224;
			int width=cv_label.cols;
			int height=cv_label.rows;
			if(width < height)
			{
				height = Dtype(rand_length) / Dtype(width) * Dtype(height);
				width = rand_length;
			}
			else
			{
				width = Dtype(rand_length) / Dtype(height) * Dtype(width);
				height = rand_length;
			}
			cv::resize(cv_label,cv_label,cv::Size(width,height),0,0,CV_INTER_CUBIC);  
			int w_off = caffe_rng_rand()%(width-crop_size);
			int h_off = caffe_rng_rand()%(height-crop_size);
			cv::Rect roi(w_off, h_off, crop_size, crop_size);
			cv_label = cv_label(roi);
		}		

		cv::resize(cv_img,cv_img,cv::Size(crop_size,crop_size),0,0,CV_INTER_CUBIC);  
		cv::resize(cv_label,cv_label,cv::Size(crop_size,crop_size),0,0,CV_INTER_CUBIC);  

		
		Dtype * prefetch_data = this->prefetch_data_.mutable_cpu_data()+ this->prefetch_data_.offset(item_id);
		for (int h = 0; h < crop_size; ++h)
    {
      const uchar* ptr = cv_img.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < crop_size; ++w)
        for (int c = 0; c < img_channels; ++c)
				{
          int top_index = (c * crop_size + h) * crop_size + w;

          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          prefetch_data[top_index] = pixel - mean_values_[c];
        }
    }
    
    Dtype * prefetch_label = this->prefetch_label_.mutable_cpu_data()+ this->prefetch_label_.offset(item_id);
		for (int h = 0; h < crop_size; ++h)
    {
      const uchar* ptr = cv_label.ptr<uchar>(h);
      int img_index = 0;
      for (int w = 0; w < crop_size; ++w)
        for (int c = 0; c < img_channels; ++c)
				{
          int top_index = (c * crop_size + h) * crop_size + w;

          Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
          prefetch_label[top_index] = pixel - mean_values_[c];
        }
    }
	

		lines_id_++;
		if (lines_id_ >= lines_size)
		{
			lines_id_ = 0;
			if (image_data_param.shuffle())
				shuffle(lines_.begin(), lines_.end());
		}
	}
	if (Solver<Dtype>::iter()%20 == 0)
	{
		lines_style_id_++;
		if (lines_style_id_ >= lines_style_size)
		{
			lines_style_id_ = 0;
			if (image_data_param.shuffle())
				shuffle(lines_style_.begin(), lines_style_.end());
		}
	}
}

template <typename Dtype>
ImageStyleDataLayer<Dtype>::~ImageStyleDataLayer<Dtype>() 
{
  if (this->thread_.get() != NULL && this->thread_->joinable())
    this->thread_->join();
}

INSTANTIATE_CLASS(ImageStyleDataLayer);
REGISTER_LAYER_CLASS(ImageStyleData);
}  // namespace caffe
