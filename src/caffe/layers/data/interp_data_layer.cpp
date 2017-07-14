
#include <vector>

#include "caffe/layers/data/interp_data_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void InterpDataLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void InterpDataLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  	int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		CHECK_EQ(bottom[0]->num(),bottom[1]->num());	
		CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());	
		CHECK_EQ(bottom[0]->height(),bottom[1]->height());	
		CHECK_EQ(bottom[0]->width(),bottom[1]->width());	
	
		top[0]->Reshape(num,channels,height,width);
}

INSTANTIATE_CLASS(InterpDataLayer);
REGISTER_LAYER_CLASS(InterpData);
}  // namespace caffe
