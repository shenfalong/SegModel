#include <vector>

#include "caffe/layers/stuff_object_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define Stuff_class 35
#define Object_class 115

namespace caffe {

template <typename Dtype>
void StuffObjectLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	stuff_mapping.Reshape(1,1,1,Stuff_class);
	object_mapping.Reshape(1,1,1,Object_class);

	FILE*  fid = fopen("stuff36.txt","r");
	if (fid == NULL)
		LOG(FATAL)<<"stuff file not found";
	for(int i=0;i<Stuff_class + Object_class;i++)
	{
		int value;
		fscanf(fid,"%d",&value);
		
		if (value < Stuff_class && value >= 0)
			stuff_mapping.mutable_cpu_data()[value] = i;	
	}
	fclose(fid);	

		
	fid = fopen("object115.txt","rb");
	if (fid == NULL)
		LOG(FATAL)<<"object file not found";
	for(int i=0;i<Stuff_class + Object_class;i++)
	{
		int value;
		fscanf(fid,"%d",&value);
		if (value < Object_class && value >= 0)
			object_mapping.mutable_cpu_data()[value] = i;	
	}	
	fclose(fid);

	LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[1]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

}

template <typename Dtype>
void StuffObjectLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{

	int num = bottom[0]->num();
	int channels_0 = bottom[0]->channels();
	int channels_1 = bottom[1]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	
	softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);	
  top[0]->Reshape(num,channels_0-1+channels_1,height,width);
	reorder_top.Reshape(num,channels_0-1+channels_1,height,width);

}

template <typename Dtype>
void StuffObjectLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void StuffObjectLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(StuffObjectLayer);
#endif

INSTANTIATE_CLASS(StuffObjectLayer);
REGISTER_LAYER_CLASS(StuffObject);
}  // namespace caffe
		
