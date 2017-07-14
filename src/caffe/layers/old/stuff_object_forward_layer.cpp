#include <vector>

#include "caffe/layers/stuff_object_forward_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	stuff_mapping.Reshape(1,1,1,35);
	object_mapping.Reshape(1,1,1,115);
	
	FILE*  fid = fopen("stuff36.txt","rb");
	if (fid == NULL)
		LOG(FATAL)<<"stuff file not found";
	for(int i=0;i<150;i++)
	{
		int value;
		fscanf(fid,"%d",&value);
		if (value < 35 && value >= 0)
			stuff_mapping.mutable_cpu_data()[value] = i - 1;	
	}
	fclose(fid);	
	fid = fopen("object115.txt","rb");
	if (fid == NULL)
		LOG(FATAL)<<"object file not found";
	for(int i=0;i<150;i++)
	{
		int value;
		fscanf(fid,"%d",&value);
		if (value < 115 && value >= 0)
			object_mapping.mutable_cpu_data()[value] = i - 1;	
	}	
	fclose(fid);
}

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  top[0]->Reshape(num,1,height,width);
  objectness.Reshape(num,1,height,width);
}

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}

template <typename Dtype>
void StuffObjectForwardLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}

#ifdef CPU_ONLY
STUB_GPU(StuffObjectForwardLayer);
#endif

INSTANTIATE_CLASS(StuffObjectForwardLayer);
REGISTER_LAYER_CLASS(StuffObjectForward);
}  // namespace caffe
		
