#include "caffe/solver.hpp"
#include <vector>
#include "caffe/util/format.hpp"

#include "caffe/layers/data/slow_style_data_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
namespace caffe {
template <typename Dtype>
static __global__ void kernel(int count, int channels,int height,int width, const Dtype *in, Dtype *out)
{

}

template <typename Dtype>
void SlowStyleDataLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}

template <typename Dtype>
void SlowStyleDataLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
#if 0
	const Dtype correction = std::sqrt(Dtype(1) - std::pow(Dtype(0.999), Solver<Dtype>::iter() + 1)) / (Dtype(1.) - std::pow(Dtype(0.9), Solver<Dtype>::iter() + 1));
	
	adam_update_gpu(top[0]->count(), top[0]->mutable_gpu_diff(), history_0_.mutable_gpu_data(),history_1_.mutable_gpu_data(), 
	Dtype(0.9), Dtype(0.999), Dtype(1e-8), correction);     
#else
	caffe_gpu_add(top[0]->count(),Dtype(1),top[0]->gpu_diff(),Dtype(0.9),history_0_.gpu_data(),top[0]->mutable_gpu_diff());
	
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),history_0_.mutable_gpu_data());
#endif	
	caffe_gpu_add(top[0]->count(),Dtype(1),top[0]->gpu_data(),Dtype(-1),top[0]->gpu_diff(),top[0]->mutable_gpu_data());
	
	if (Solver<Dtype>::iter() % 500 == 0)
 	{
		LOG(INFO)<<"---------------writing image-----------------";
		std::vector<float> mean_values_;
		mean_values_.clear();
		mean_values_.resize(3);
		mean_values_[0] = 104.008;
	  mean_values_[1] = 116.669;
	  mean_values_[2] = 122.675;
		int num = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		const Dtype * top_data = top[0]->cpu_data();
		cv::Mat cv_im(height*1,width*1,CV_8UC3);
		//cv::Mat cv_im(height*8,width*8,CV_8UC1);
		for (int i=0;i<1*height;i++)
		{
			unsigned char * data_ptr = cv_im.ptr<uchar>(i);
			for (int j=0;j<1*width;j++)						
			{
				for (int c=0;c<channels;c++)
				{				
					int n = (i/height)*1+(j/width);
					int h = i%height;
					int w = j%width;
					int index = ((n*channels+c)*height+h)*width+w;
					data_ptr[j*channels+c] = min(max(top_data[index] + mean_values_[c],Dtype(0)),Dtype(255));		
				}
			}
		}
		std::stringstream ss;
		string filename;
		int gpu_id_;
		CUDA_CHECK(cudaGetDevice(&gpu_id_));
		ss<<"generateimage//"<<Solver<Dtype>::iter()<<"GPU"<<gpu_id_<<".jpg";
		ss>>filename;
		cv::imwrite(filename,cv_im);
	}
}
template <typename Dtype>
void SlowStyleDataLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

}
INSTANTIATE_LAYER_GPU_FUNCS(SlowStyleDataLayer);
}  // namespace caffe
