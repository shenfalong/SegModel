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

#include "caffe/layers/data/image_face_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
void ImageFaceDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  TransformationParameter transform_param = this->layer_param_.transform_param();
  DataParameter image_data_param    = this->layer_param_.data_param();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = transform_param.crop_size();

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
    imgfn = imgfn + ".jpg";
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  if (image_data_param.shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines_.begin(), lines_.end());
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;


  this->prefetch_data_.Reshape(batch_size, 3, crop_size, crop_size);		  		  
  this->prefetch_label_.Reshape(batch_size, 1, 1, 1);		 

}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageFaceDataLayer<Dtype>::InternalThreadEntry(int gpu_id)
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
	
	int rand_number = 1;//caffe_rng_rand()%11+1;
  for(int item_id = 0; item_id < batch_size; item_id++)
  {
		CHECK_GT(lines_size, lines_id_);
		cv::Mat cv_img;
		cv_img = cv::imread(root_folder + lines_[lines_id_].first, CV_LOAD_IMAGE_COLOR);
		if (!cv_img.data) {LOG(FATAL) << "Fail to load img: " << root_folder + lines_[lines_id_].first;}
		int img_channels = cv_img.channels();
		CHECK_EQ(img_channels,3);

	  //int w_off = 25;
	  //int h_off = 50;
		//cv::Rect roi(w_off, h_off, 128, 128);
    //cv_img = cv_img(roi);
    if (cv_img.rows < 128 || cv_img.cols < 128)
    {
    	lines_id_++;
			if (lines_id_ >= lines_size)
			{
				lines_id_ = 0;
				if (image_data_param.shuffle())
					shuffle(lines_.begin(), lines_.end());
			}
			item_id--;
			continue;
    }
   	
   	if (cv_img.rows < cv_img.cols)
   	{
		  int resize_side = cv_img.cols *  Dtype(crop_size) / Dtype(cv_img.rows);
			cv::resize(cv_img,cv_img,cv::Size(resize_side,crop_size),0,0,CV_INTER_CUBIC);
			int w_off = (resize_side - crop_size) / 2;
		  cv::Rect roi(w_off, 0, crop_size, crop_size);
		  cv_img = cv_img(roi);
		}
		else
		{
			int resize_side = cv_img.rows * Dtype(crop_size) / Dtype(cv_img.cols);
			cv::resize(cv_img,cv_img,cv::Size(crop_size,resize_side),0,0,CV_INTER_CUBIC);
			int h_off = (resize_side - crop_size) / 2;
		  cv::Rect roi(0, h_off, crop_size, crop_size);
		  cv_img = cv_img(roi);
		}
		
		
		
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
          prefetch_data[top_index] = (pixel - Dtype(127.5) )/Dtype(127.5);
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
}

template <typename Dtype>
ImageFaceDataLayer<Dtype>::~ImageFaceDataLayer<Dtype>() 
{
  if (this->thread_.get() != NULL && this->thread_->joinable())
    this->thread_->join();
}

INSTANTIATE_CLASS(ImageFaceDataLayer);
REGISTER_LAYER_CLASS(ImageFaceData);
}  // namespace caffe
