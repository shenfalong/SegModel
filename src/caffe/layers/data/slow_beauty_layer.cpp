#include "caffe/solver.hpp"
#include <vector>
#include "caffe/util/format.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/data/slow_beauty_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SlowBeautyLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

	int crop_size = 128;
	int img_channels = 3;
	top[0]->Reshape(1,128,1,1);
	top[1]->Reshape(1,img_channels,crop_size,crop_size);

	
	history_0_.Reshape(1,128,1,1);
	history_1_.Reshape(1,128,1,1);
	caffe_rng_uniform<Dtype>(top[0]->count(), Dtype(-1), Dtype(1), top[0]->mutable_cpu_data());

	
	
	std::vector<float> mean_values_;
	mean_values_.clear();
	mean_values_.resize(3);
	mean_values_[0] = 0;
  mean_values_[1] = 0;
  mean_values_[2] = 0;
	
	cv::Mat cv_img = cv::imread("reference.jpg");
	cv::resize(cv_img,cv_img,cv::Size(crop_size,crop_size),0,0,CV_INTER_CUBIC);  

	
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
        prefetch_data[top_index] = (pixel - 127.5)/127.5;
      }
  }
}

template <typename Dtype>
void SlowBeautyLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}

INSTANTIATE_CLASS(SlowBeautyLayer);
REGISTER_LAYER_CLASS(SlowBeauty);
}  // namespace caffe
