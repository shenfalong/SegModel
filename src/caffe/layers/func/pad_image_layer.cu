
#include <vector>

#include "caffe/layers/func/pad_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
template <typename Dtype>
static __global__ void pad_kernel(int count, int height,int width, int pad, const Dtype *in, Dtype *out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int nc = i / (width+pad*2) / (height+pad*2) ;
		int h = i / (width+pad*2) % (height+pad*2);
		int w = i % (width+pad*2);
		if (h < pad)
			h = pad - 1 - h;
		else if (h < pad+height)
			h = h - pad;
		else
			h = height - 1 - (h - (pad+height));
		if (w < pad)
			w = pad - 1 - w;
		else if (w < pad+width)
			w = w - pad;
		else
			w = width - 1 - (w - (pad+width));
		out[i] = in[(nc*height+h)*width+w];			 
	}
}

template <typename Dtype>
void PadImageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	pad_kernel<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),height,width, pad_, bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
#if 0
FILE *fid = fopen("debug","wb");
fwrite(top[0]->cpu_data(),sizeof(Dtype),top[0]->count(),fid);
fclose(fid);
LOG(FATAL)<<height<<", "<<width;
#endif		
	
}

template <typename Dtype>
void PadImageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	
}
template <typename Dtype>
void PadImageLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(PadImageLayer);
}  // namespace caffe
