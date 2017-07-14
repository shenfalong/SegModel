#include "caffe/solver.hpp"
#include <vector>
#include "caffe/util/format.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/data/slow_style_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SlowStyleDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	int crop_size = 256;
	int img_channels = 3;
	top[0]->Reshape(1,img_channels,crop_size,crop_size);
	top[1]->Reshape(1,img_channels,crop_size,crop_size);
	top[2]->Reshape(1,img_channels,crop_size,crop_size);
	
	history_0_.Reshape(1,img_channels,crop_size,crop_size);
	history_1_.Reshape(1,img_channels,crop_size,crop_size);
	
	caffe_rng_gaussian<Dtype>(top[0]->count(), 
    													Dtype(0), Dtype(1), 
    													top[0]->mutable_cpu_data());
	caffe_gpu_scal(top[0]->count(), Dtype(127.5), top[0]->mutable_gpu_data());
	caffe_gpu_add_scalar(top[0]->count(), Dtype(127.5), top[0]->mutable_gpu_data());
	
	
	std::vector<float> mean_values_;
	mean_values_.clear();
	mean_values_.resize(3);
	mean_values_[0] = 104.008;
  mean_values_[1] = 116.669;
  mean_values_[2] = 122.675;
	
	cv::Mat cv_img = cv::imread("content.jpg");
	cv::Mat cv_label = cv::imread("style.jpg");
	cv::resize(cv_img,cv_img,cv::Size(crop_size,crop_size),0,0,CV_INTER_CUBIC);  
	cv::resize(cv_label,cv_label,cv::Size(crop_size,crop_size),0,0,CV_INTER_CUBIC);  

	
	Dtype * prefetch_data = top[1]->mutable_cpu_data();
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
  
  Dtype * prefetch_label = top[2]->mutable_cpu_data();
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
}

template <typename Dtype>
void SlowStyleDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	
}


INSTANTIATE_CLASS(SlowStyleDataLayer);
REGISTER_LAYER_CLASS(SlowStyleData);
}  // namespace caffe
