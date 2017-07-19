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

#include "caffe/layers/data/image_class_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
void ImageClassDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
  TransformationParameter transform_param = this->layer_param_.transform_param();
  DataParameter image_data_param    = this->layer_param_.data_param();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = transform_param.crop_size();


  // Read the file with filenames and labels
  const string& source = image_data_param.source(0);
  LOG(INFO) << "Opening file " << source;
  
  lines_.resize(10);
  lines_id_.resize(10);

  for (int i=0;i<10;i++)
  {
  	string filename = source.c_str()+format_int(i)+".txt";
		std::ifstream infile(filename.c_str());
		string linestr;
		lines_[i].clear();
		while (std::getline(infile, linestr))
		{
		  std::istringstream iss(linestr);
		  string imgfn;
		  iss >> imgfn;
		  lines_[i].push_back(std::make_pair(imgfn, imgfn));
		}

		if (image_data_param.shuffle()) {
		  LOG(INFO) << "Shuffling data";
		  shuffle(lines_[i].begin(), lines_[i].end());
		}
		LOG(INFO) << "A total of " << lines_[i].size() << " images.";
		lines_id_[i] = 0;
	}

  this->prefetch_data_.Reshape(batch_size, 3, crop_size, crop_size);		  		  
  this->prefetch_label_.Reshape(batch_size, 10, 1, 1);		 

	cur_class = 0;
}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageClassDataLayer<Dtype>::InternalThreadEntry(int gpu_id)
{
 //new thread treat GPU 0 as default device, so it is necessary to set device in case of
  //ghost memory
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[gpu_id]));

	
 	TransformationParameter transform_param = this->layer_param_.transform_param();
  DataParameter image_data_param    = this->layer_param_.data_param();
  const int batch_size = image_data_param.batch_size();
  const int crop_size = transform_param.crop_size();
  string root_folder   = image_data_param.root_folder();
	
	
	//if (Solver<Dtype>::iter()%20 == 0)
	//	cur_class = caffe_rng_rand() % 10;
	

	caffe_set(this->prefetch_label_.count(),Dtype(0),this->prefetch_label_.mutable_cpu_data());

  for(int item_id = 0; item_id < batch_size; item_id++)
  {
		cv::Mat cv_img = cv::imread(root_folder + format_int(cur_class) + "/" + lines_[cur_class][lines_id_[cur_class]].first, CV_LOAD_IMAGE_COLOR);
		if (!cv_img.data) {LOG(FATAL) << "Fail to load img: " << root_folder + format_int(cur_class) + "/" + lines_[cur_class][lines_id_[cur_class]].first;}
		int img_channels = cv_img.channels();
	
		cv::resize(cv_img,cv_img,cv::Size(cv_img.cols+8,cv_img.rows+8),0,0,CV_INTER_CUBIC);
		int h_off = caffe_rng_rand()%(cv_img.rows - crop_size + 1);
    int w_off = caffe_rng_rand()%(cv_img.cols - crop_size + 1);
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_img = cv_img(roi);
		
	
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
          prefetch_data[top_index] = (pixel - 127.5)/127.5;
        }
    }
    this->prefetch_label_.mutable_cpu_data()[item_id*10+cur_class] = 1;
    
    lines_id_[cur_class]++;
    if (lines_id_[cur_class] >= lines_[cur_class].size())
    {
      lines_id_[cur_class] = 0;
      if (image_data_param.shuffle())
        shuffle(lines_[cur_class].begin(), lines_[cur_class].end());
    }
	}
}

template <typename Dtype>
ImageClassDataLayer<Dtype>::~ImageClassDataLayer<Dtype>() 
{
  if (this->thread_.get() != NULL && this->thread_->joinable())
    this->thread_->join();
}

INSTANTIATE_CLASS(ImageClassDataLayer);
REGISTER_LAYER_CLASS(ImageClassData);
}  // namespace caffe
