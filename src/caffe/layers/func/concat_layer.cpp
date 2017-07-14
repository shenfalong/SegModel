#include <vector>

#include "caffe/layers/func/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void ConcatLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{

}

template <typename Dtype>
void ConcatLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	if (bottom.size() == 2)
	{
		const int num = bottom[0]->num();
		const int channels0 = bottom[0]->channels();
		const int channels1 = bottom[1]->channels();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
	 	
	 	CHECK_EQ(bottom[0]->num(),bottom[1]->num());

	 	CHECK_EQ(bottom[0]->width(),bottom[1]->width());
 	
	 	CHECK_EQ(bottom[0]->height(),bottom[1]->height());
		
		top[0]->Reshape(num,channels0+channels1,height,width);
	}
	else if (bottom.size() > 2)
	{
		const int num = bottom[0]->num();
		const int height = bottom[0]->height();
		const int width = bottom[0]->width();
	 	
	 	int channels = 0;
	 	for (int i=0;i<bottom.size();i++)
	 	{
		 	CHECK_EQ(bottom[0]->num(),bottom[i]->num()); 	
		 	CHECK_EQ(bottom[0]->width(),bottom[i]->width());
		 	CHECK_EQ(bottom[0]->height(),bottom[i]->height());
			channels += bottom[i]->channels();
		}
		
		top[0]->Reshape(num,channels,height,width);
	}
	else
   LOG(FATAL)<<"unspported size";
}



INSTANTIATE_CLASS(ConcatLayer);
REGISTER_LAYER_CLASS(Concat);
}  // namespace caffe
