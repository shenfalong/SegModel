
#include <vector>

#include "caffe/layers/func/de_concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DeConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void DeConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
 	
 	CHECK_EQ(this->layer_param_.concat_param().channels_size(),top.size());
 	
 	int all_channels = 0;
 	for (int i=0;i<top.size();i++)
 	{
 		int i_channels = this->layer_param_.concat_param().channels(i);
	 	top[i]->Reshape(num,i_channels,height,width);
	 	all_channels += i_channels;
	}
	
	CHECK_EQ(channels,all_channels);

}

#ifdef CPU_ONLY
STUB_GPU(DeConcatLayer);
#endif

INSTANTIATE_CLASS(DeConcatLayer);
REGISTER_LAYER_CLASS(DeConcat);
}  // namespace caffe
