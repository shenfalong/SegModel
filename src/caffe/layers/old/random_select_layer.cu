
#include <vector>

#include "caffe/layers/random_select_layer.hpp"
#include "caffe/util/math_functions.hpp"

//----------------------------------------------------------------
namespace caffe {
template <typename Dtype>
static __global__ void signed_forward_kernel(int count,const Dtype *a,Dtype *b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
	#if 0
		if (a[i]>1)
			b[i] = 2*sqrt(a[i])-1;
		else if(a[i]>-1)
			b[i] = a[i];
		else
			b[i] = -2*sqrt(-a[i])+1;
	#endif
		b[i] = a[i];
	}
}
template <typename Dtype>
static __global__ void signed_backward_kernel(int count,const Dtype *b_diff,const Dtype * b_data,Dtype *a_diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{
	#if 0
		if (b_data[i]>1)
			a_diff[i]=b_diff[i] * 2 / (b_data[i]+1);
		else if(b_data[i]>-1)
			a_diff[i]=b_diff[i];
		else
			a_diff[i]=b_diff[i] * 2 / (-b_data[i]+1);
	#endif
		a_diff[i] = b_diff[i];
	}
}
template <typename Dtype>
static __global__ void randomized_forward_0_(int count,int channels,int height,int width,int stride,const Dtype *a, const Dtype *rand_choose,Dtype *b)
{

	CUDA_KERNEL_LOOP(i, count)
	{

		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
			
		b[i] = 0;
//--------------------------------------------------------------------------------------------
		if (rand_choose[c] == 0) {
			if (h > 0 && w < width -1)	
				b[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* a[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride+1];
		}
//--------------------------------------------------------------------------------------------			
		else if (rand_choose[c] == 1) {
			if (w < width - 1)		
				b[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride+1];
		}
//--------------------------------------------------------------------------------------------			
		else if (rand_choose[c] == 2) {	
			if (h < height - 1)
				b[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* a[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride];
		}
//--------------------------------------------------------------------------------------------			
		else {	
			if (h < height -1 && w < width - 1)
				b[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* a[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride+1];	
		}	
//--------------------------------------------------------------------------------------------			
	}

}
template <typename Dtype>
static __global__ void randomized_backward_0_(int count,int channels,int height,int width, int stride, const Dtype *out_diff, const Dtype * in_data, const Dtype *rand_choose, Dtype * in_diff)
{

	CUDA_KERNEL_LOOP(i, count)
	{

		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		if (stride == 1)
		{
			Dtype sum = 0;
	//--------------------------------------------------------------------------------------------	
			if (rand_choose[c] == 0) {
				if (h > 0 && w < width - 1)
					sum += out_diff[((n*channels+c)*height+h)*width+w] * in_data[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride+1];	
				if (h < height-1 && w > 0)
					sum += out_diff[((n*channels+c)*height+h+1)*width+w-1] * in_data[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride-1];	
			}	
	//--------------------------------------------------------------------------------------------	
			else if (rand_choose[c] == 1) {
				if (w < width - 1)
					sum += out_diff[((n*channels+c)*height+h)*width+w] * in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride+1];
				if (w > 0)	
					sum += out_diff[((n*channels+c)*height+h)*width+w-1] * in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride-1];
			}
	//--------------------------------------------------------------------------------------------	
			else if (rand_choose[c] == 2) {				
				if (h < height - 1)
					sum += out_diff[((n*channels+c)*height+h)*width+w] * in_data[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride];	
				if (h > 0)	
					sum += out_diff[((n*channels+c)*height+h-1)*width+w] * in_data[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride];	
			}	
	//--------------------------------------------------------------------------------------------	
			else {
				if (h < height - 1 && w < width - 1)
					sum += out_diff[((n*channels+c)*height+h)*width+w] * in_data[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride+1];
				if (h > 0 && w > 0)
					sum += out_diff[((n*channels+c)*height+h-1)*width+w-1] * in_data[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride-1];
			}
	//--------------------------------------------------------------------------------------------				
			in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += sum;	
		}
		else if (stride == 2)
		{
	//--------------------------------------------------------------------------------------------	
			if (rand_choose[c] == 0) {
				if (h > 0 && w < width -1)	
				{
					in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += out_diff[((n*channels+c)*height+h)*width+w] 
																																				* in_data[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride+1];	
					in_diff[((n*channels+c)*height*stride+h*stride-1)*width*stride+w*stride+1] += out_diff[((n*channels+c)*height+h)*width+w]  
																																				* in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride];	
				}
			}	
	//--------------------------------------------------------------------------------------------	
			else if (rand_choose[c] == 1) {
				if (w < width - 1)
				{
					in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += out_diff[((n*channels+c)*height+h)*width+w] 
																																				* in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride+1];	
					in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride+1] += out_diff[((n*channels+c)*height+h)*width+w]  
																																			* in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride];	
				}																																																																		
			}
	//--------------------------------------------------------------------------------------------	
			else if (rand_choose[c] == 2) {				
				if (h < height - 1)
				{
					in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += out_diff[((n*channels+c)*height+h)*width+w] 
																																				* in_data[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride];
					in_diff[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride] += out_diff[((n*channels+c)*height+h)*width+w]  
																															          * in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride];
				}
			}	
	//--------------------------------------------------------------------------------------------	
			else {
				if (h < height - 1 && w < width - 1)
				{
					in_diff[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += out_diff[((n*channels+c)*height+h)*width+w] 
																																				* in_data[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride+1];
					in_diff[((n*channels+c)*height*stride+h*stride+1)*width*stride+w*stride+1] += out_diff[((n*channels+c)*height+h)*width+w]  
																															          * in_data[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride];			
				}
			}
	//--------------------------------------------------------------------------------------------					
		}
		else
			return;
	}

}
//----------------------------------------------------------------
template <typename Dtype>
static __global__ void randomized_forward_1_(int count,int channels,int height,int width, int stride,const Dtype *a, const Dtype *b, const Dtype *rand_choose,Dtype *cc)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		
		cc[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* b[((n*channels+c)*height+h)*width+w];
#if 0
		cc[i] = 0;
//--------------------------------------------------------------------------------------------
		if (rand_choose[c] == 0) {
			if (h > 0 && w < width -1)	
				cc[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* b[((n*channels+c)*height+h-1)*width+w+1];
		}
//--------------------------------------------------------------------------------------------			
		else if (rand_choose[c] == 1) {
			if (w < width - 1)		
				cc[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* b[((n*channels+c)*height+h)*width+w+1];
		}
//--------------------------------------------------------------------------------------------			
		else if (rand_choose[c] == 2) {	
			if (h < height - 1)
				cc[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* b[((n*channels+c)*height+h+1)*width+w];
		}
//--------------------------------------------------------------------------------------------			
		else {	
			if (h < height -1 && w < width - 1)
				cc[i] = a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride]* b[((n*channels+c)*height+h+1)*width+w+1];	
		}	
//--------------------------------------------------------------------------------------------	
#endif		
	}
}
template <typename Dtype>
static __global__ void randomized_backward_1_(int count,int channels,int height,int width, int stride, const Dtype *diff_c, const Dtype *a, const Dtype *b, const Dtype *rand_choose, Dtype * diff_a,Dtype * diff_b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		diff_a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += diff_c[i] * b[i];
		diff_b[i]                                                             += diff_c[i] * a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride];	
#if 0		
		Dtype sum_a = 0;
		Dtype sum_b = 0;
//--------------------------------------------------------------------------------------------	
		if (rand_choose[c] == 0) {
			if (h > 0 && w < width - 1)
				sum_a += diff_c[i] * b[((n*channels+c)*height+h-1)*width+w+1];	
			if (h < height-1 && w > 0)
				sum_b += diff_c[((n*channels+c)*height+h+1)*width+w-1] * a[((n*channels+c)*height*stride+(h+1)*stride)*width*stride+(w-1)*stride];	
		}	
//--------------------------------------------------------------------------------------------	
		else if (rand_choose[c] == 1) {
			if (w < width - 1)
				sum_a += diff_c[i] * b[((n*channels+c)*height+h)*width+w+1];
			if (w > 0)	
				sum_b += diff_c[((n*channels+c)*height+h)*width+w-1] * a[((n*channels+c)*height*stride+h*stride)*width*stride+(w-1)*stride];
		}
//--------------------------------------------------------------------------------------------	
		else if (rand_choose[c] == 2) {				
			if (h < height - 1)
				sum_a += diff_c[i] * b[((n*channels+c)*height+h+1)*width+w];	
			if (h > 0)	
				sum_b += diff_c[((n*channels+c)*height+h-1)*width+w] * a[((n*channels+c)*height*stride+(h-1)*stride)*width*stride+w*stride];	
		}	
//--------------------------------------------------------------------------------------------	
		else {
			if (h < height - 1 && w < width - 1)
				sum_a += diff_c[i] * b[((n*channels+c)*height+h+1)*width+w+1];
			if (h > 0 && w > 0)
				sum_b += diff_c[((n*channels+c)*height+h-1)*width+w-1] * a[((n*channels+c)*height*stride+(h-1)*stride)*width*stride+(w-1)*stride];
		}
//--------------------------------------------------------------------------------------------				
		diff_a[((n*channels+c)*height*stride+h*stride)*width*stride+w*stride] += sum_a;
		diff_b[i] += sum_b;	
#endif
	}
}
//----------------------------------------------------
template <typename Dtype>
static __global__ void concat_forward(int count,int channels,int height,int width,
																const Dtype *a0, const Dtype * a1, Dtype *b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / (channels*2);
		int c = i / width / height % (channels*2);
		int h = i / width % height;
		int w = i % width;
		
		if (c<channels)
			b[i] = a0[((n*channels+c         )*height+h)*width+w];
		else 
			b[i] = a1[((n*channels+c-channels)*height+h)*width+w];
		
			
	}
}
template <typename Dtype>
static __global__ void concat_backward(int count,int channels,int height,int width,
																const Dtype * b, Dtype *a0,  Dtype * a1)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / (channels*2);
		int c = i / width / height % (channels*2);
		int h = i / width % height;
		int w = i % width;
		
		if (c<channels)
			a0[((n*channels+c         )*height+h)*width+w] = b[i];
		else
			a1[((n*channels+c-channels)*height+h)*width+w] = b[i];
			
	}
}
//---------------------------------------------------
template <typename Dtype>
void RandomSelectLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height()/stride_;
	int width = bottom[0]->width()/stride_;
	switch (this->layer_param_.sec_param().sec_feature()) 
	{
		case SecParameter_SecFeature_SEC_0:
		{
			conv3x3_layer_0_->Forward_gpu(conv3x3_bottom_vec_0_, conv3x3_top_vec_0_);//bottom_buffer_
			conv3x3_layer_1_->Forward_gpu(conv3x3_bottom_vec_1_, conv3x3_top_vec_1_);//top_buffer_
			
			randomized_forward_1_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width,stride_,
									bottom[0]->gpu_data(),bottom_buffer_->gpu_data(),this->blobs_[0]->gpu_data(),sec_buffer_->mutable_gpu_data());			
			//caffe_copy(bottom[0]->count(),bottom_buffer_->gpu_data(),sec_buffer_->mutable_gpu_data());
			
			
			concat_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width, 
									sec_buffer_->gpu_data(),top_buffer_->gpu_data(),top[0]->mutable_gpu_data());			
		}
		break;
		case SecParameter_SecFeature_SEC_1:
		{			
			randomized_forward_0_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width,stride_,
							bottom[0]->gpu_data(),this->blobs_[0]->gpu_data(),top_buffer_->mutable_gpu_data());
					
			conv3x3_layer_0_->Forward_gpu(conv3x3_bottom_vec_0_, conv3x3_top_vec_0_);
			
			concat_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width,
							bottom_buffer_->gpu_data(),top_buffer_->gpu_data(),top[0]->mutable_gpu_data());
		}
		break;
		case SecParameter_SecFeature_SEC_2:
		{
			conv3x3_layer_0_->Forward_gpu(conv3x3_bottom_vec_0_, conv3x3_top_vec_0_);
			//caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),bottom_buffer_->mutable_gpu_data());
			
			randomized_forward_0_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width,1,
							bottom_buffer_->gpu_data(),this->blobs_[0]->gpu_data(),top_buffer_->mutable_gpu_data());
		
										
			concat_forward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width,
							bottom_buffer_->gpu_data(),top_buffer_->gpu_data(),top[0]->mutable_gpu_data());					
		}		
		break;
		default:
			LOG(FATAL)<<"unknow random select option";
	}
}

template <typename Dtype>
void RandomSelectLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height()/stride_;
	int width = bottom[0]->width()/stride_;
	switch (this->layer_param_.sec_param().sec_feature()) 
	{
		case SecParameter_SecFeature_SEC_0:
		{
			concat_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width, top[0]->gpu_diff(), 
													sec_buffer_->mutable_gpu_diff(),top_buffer_->mutable_gpu_diff());	
			
			caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
			caffe_gpu_set(bottom_buffer_->count(),Dtype(0),bottom_buffer_->mutable_gpu_diff());
			randomized_backward_1_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width,stride_,
							sec_buffer_->gpu_diff(),bottom[0]->gpu_data(),bottom_buffer_->gpu_data(),this->blobs_[0]->gpu_data(), 
							bottom[0]->mutable_gpu_diff(),bottom_buffer_->mutable_gpu_diff());	
			//caffe_copy(bottom[0]->count(),sec_buffer_->gpu_diff(),bottom_buffer_->mutable_gpu_diff());
			//caffe_gpu_set(bottom[0]->count(),Dtype(0),bottom[0]->mutable_gpu_diff());
			
			sec_buffer_->ReshapeLike(*bottom[0]);
			caffe_copy(sec_buffer_->count(),bottom[0]->gpu_diff(),sec_buffer_->mutable_gpu_diff());
			
			conv3x3_layer_1_->Backward_gpu(conv3x3_top_vec_1_, conv3x3_bottom_vec_1_);			
			caffe_gpu_add(sec_buffer_->count(),Dtype(1),sec_buffer_->gpu_diff(),Dtype(1),bottom[0]->gpu_diff(),sec_buffer_->mutable_gpu_diff());
			
			conv3x3_layer_0_->Backward_gpu(conv3x3_top_vec_0_, conv3x3_bottom_vec_0_);								
			caffe_gpu_add(sec_buffer_->count(),Dtype(1),sec_buffer_->gpu_diff(),Dtype(1),bottom[0]->gpu_diff(),sec_buffer_->mutable_gpu_diff());
			
			caffe_copy(bottom[0]->count(),sec_buffer_->gpu_diff(),bottom[0]->mutable_gpu_diff());
		}
		break;
		case SecParameter_SecFeature_SEC_1:
		{
			concat_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width,
									top[0]->gpu_diff(), bottom_buffer_->mutable_gpu_diff(),top_buffer_->mutable_gpu_diff());

			conv3x3_layer_0_->Backward_gpu(conv3x3_top_vec_0_, conv3x3_bottom_vec_0_);

			randomized_backward_0_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width, stride_,
									top_buffer_->gpu_diff(),bottom[0]->gpu_data(),this->blobs_[0]->gpu_data(),bottom[0]->mutable_gpu_diff());	
		}
		break;
		case SecParameter_SecFeature_SEC_2:
		{
			concat_backward<Dtype><<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top[0]->count(),channels,height,width,
									top[0]->gpu_diff(), bottom_buffer_->mutable_gpu_diff(),top_buffer_->mutable_gpu_diff());
	
						
			randomized_backward_0_<Dtype><<<CAFFE_GET_BLOCKS(top_buffer_->count()), CAFFE_CUDA_NUM_THREADS>>>
			(top_buffer_->count(),channels,height,width,1,
									top_buffer_->gpu_diff(),bottom_buffer_->gpu_data(),this->blobs_[0]->gpu_data(),bottom_buffer_->mutable_gpu_diff());	
			
			conv3x3_layer_0_->Backward_gpu(conv3x3_top_vec_0_, conv3x3_bottom_vec_0_);		
			//caffe_copy(bottom[0]->count(),bottom_buffer_->gpu_diff(),bottom[0]->mutable_gpu_diff());	
		}		
		break;
		default:
			LOG(FATAL)<<"unknow random select option";
	}
}
INSTANTIATE_LAYER_GPU_FUNCS(RandomSelectLayer);
}  // namespace caffe
