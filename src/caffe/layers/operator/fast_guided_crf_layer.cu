#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/layers/operator/fast_guided_crf_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
static __global__ void subsampling(const int num,const int channels,const int height_in,const int width_in,const int height_out,const int width_out,const Dtype * id, Dtype *od)
{
	CUDA_KERNEL_LOOP(i, num*channels*height_out*width_out)
  {
  	int w = i % width_out;
  	int h = i / width_out % height_out;
  	int c = i / width_out / height_out % channels;
  	int n = i / width_out / height_out / channels;
  		
  	od[i] = id[((n*channels+c)*height_in+h*4)*width_in+w*4];
  }
}
template <typename Dtype>
static __global__ void interp(const int num, const int channels,const int height_in,const int width_in,const int height_out,const int width_out,const Dtype * id, Dtype *od)
{
	CUDA_KERNEL_LOOP(i, num*channels*height_out*width_out)
  {
  	int w = i % width_out;
  	int h = i / width_out % height_out;
  	int c = i / width_out / height_out % channels;
  	int n = i / width_out / height_out / channels;
  		
  	od[i] = id[((n*channels+c)*height_in+h/4)*width_in+w/4];
  }
}

template <typename Dtype>
static __global__ void softmax_forward_kernel(const int maxStates,const int nNodes, const Dtype * energy,Dtype * prob)
{
	CUDA_KERNEL_LOOP(n, nNodes)
	{
		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] = energy[s*nNodes+n];

		Dtype max_prob = Dtype(-FLT_MAX);
		for(int s=0;s<maxStates;s++)
			max_prob =max(max_prob,prob[s*nNodes+n]);

		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] -= max_prob;

		Dtype sum = 0;
		for(int s=0;s<maxStates;s++)
			sum += exp(prob[s*nNodes+n]);

		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] = exp(prob[s*nNodes+n]) / sum;
	}
}
template <typename Dtype>
static __global__ void softmax_backward_kernel(const int maxStates,const int nNodes, const Dtype * top_diff,const Dtype *prob,Dtype * bottom_diff)
{
	CUDA_KERNEL_LOOP(ind, nNodes*maxStates)
	{
		int n=ind % nNodes;
		int s=ind / nNodes;
		for(int s2=0;s2<maxStates;s2++)
			bottom_diff[s*nNodes+n] += top_diff[s2*nNodes+n]*prob[s2*nNodes+n]*(Dtype(s==s2)-prob[s*nNodes+n]);
	}
}
//--------------------------------------------------------------
template <typename Dtype>
static __global__ void vector_product_kernel(const int num,const int channels1,const int channels2, const int spatial_dim,const Dtype * a,const Dtype * b,Dtype *var)//var = a .* b
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*channels1*channels2*num)
	{	
		int n   = ind / spatial_dim / channels1 / channels2;
		int c2  = ind / spatial_dim / channels1 % channels2;
		int c1  = ind / spatial_dim % channels1;
		int s   = ind % spatial_dim;
		
		
		var[ind]=a[(n*channels1+c1)*spatial_dim+s]*b[(n*channels2+c2)*spatial_dim+s];
	}
}
template <typename Dtype>
static __global__ void substract_vector_product_kernel(const int num, const int channels1,const int channels2,const int spatial_dim,const Dtype *avg,const Dtype *a,const Dtype *b, Dtype * var)//var = avg - a.*b;
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*channels1*channels2*num)
	{
		int n  = ind / spatial_dim / channels1 / channels2;
		int c2 = ind / spatial_dim / channels1 % channels2;
		int c1 = ind / spatial_dim % channels1;	
		int s  = ind % spatial_dim;
		var[ind]=avg[ind]-a[(n*channels1+c1)*spatial_dim+s]*b[(n*channels2+c2)*spatial_dim+s];
	}
}
template <typename Dtype>
static __global__ void inv_var_I_eps_kernel_3(const int num, const int channels, const int spatial_dim, const Dtype eps,Dtype *var_I,Dtype *inv_var_I)
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*num)
	{
		int n = ind / spatial_dim;
		int s = ind % spatial_dim;
		
		for(int c=0;c<channels;c++)
			var_I[(n*channels*channels+(c*channels+c))*spatial_dim+s]=var_I[(n*channels*channels+(c*channels+c))*spatial_dim+s]+eps;

		Dtype det = var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s])
				- var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s])
				+ var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+0*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+0*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+0*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+1*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+1*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+1*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+2*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+2*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+2*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]);
	}

}
template <typename Dtype>
static __global__ void div_sum_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const Dtype *inv_var_I,const Dtype *cov_Ip,
																 Dtype *a)
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		
		a[((n*maxStates+m)*channels+0)*spatial_dim+s] = cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+2)*spatial_dim+s];

		a[((n*maxStates+m)*channels+1)*spatial_dim+s]	= cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]
																	  + cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+2)*spatial_dim+s];

		a[((n*maxStates+m)*channels+2)*spatial_dim+s] = cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+2)*spatial_dim+s];
	}
}
template <typename Dtype>
static __global__ void substract_vector_matrix_product_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const Dtype * mean_p,const Dtype * a,const Dtype * mean_I,Dtype *b)//	b = mean_p - mean_I *. a;
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		b[ind] = mean_p[ind]
				   - mean_I[(n*3+0)*spatial_dim+s] * a[((n*maxStates+m)*channels+0)*spatial_dim+s]
				   - mean_I[(n*3+1)*spatial_dim+s] * a[((n*maxStates+m)*channels+1)*spatial_dim+s]
				   - mean_I[(n*3+2)*spatial_dim+s] * a[((n*maxStates+m)*channels+2)*spatial_dim+s];
	}
}
template <typename Dtype>
static __global__ void vector_matrix_product_sum_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const Dtype *mean_a,const Dtype *I,const Dtype *mean_b,Dtype *q)// q = I .* mean_a + mean_b;
{

	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		q[ind] = I[(n*3+0)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+0)*spatial_dim+s]
					 + I[(n*3+1)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+1)*spatial_dim+s]
				   + I[(n*3+2)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+2)*spatial_dim+s]
				   + mean_b[ind];
	}

}
//---------------------------------------------
template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::guided_filter_gpu(const int num,const int channels,const int maxStates,const int height,const int width,const Dtype *I,const Dtype *I_sub,const Dtype * p,Dtype * p_sub,Dtype *output_p)
{
	const int spatial_dim=height*width;
	
	subsampling<Dtype><<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
  	(num,maxStates,height,width,height/4,width/4,p,p_sub);
  	
  	
	//******************************** prob ************************************
	box_filter_gpu(num,maxStates,height/4,width/4,radius,p_sub,mean_p.mutable_gpu_data(),buffer_score.mutable_gpu_data());

	vector_product_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*maxStates*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim/4/4,I_sub,p_sub,Ip.mutable_gpu_data());//Ip = I .* p;
	
	box_filter_gpu(num,channels*maxStates,height/4,width/4,radius,Ip.gpu_data(),mean_Ip.mutable_gpu_data(),buffer_image_score.mutable_gpu_data());


	substract_vector_product_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*maxStates*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim/4/4,mean_Ip.gpu_data(),mean_I.gpu_data(),mean_p.gpu_data(), cov_Ip.mutable_gpu_data());//cov_Ip = mean_Ip - mean_I .* mean_p;


	inv_var_I_eps_kernel_3<Dtype><<<CAFFE_GET_BLOCKS(num*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,spatial_dim/4/4,eps,var_I.mutable_gpu_data(),inv_var_I.mutable_gpu_data());//inv_var_I=inv(var_I + eps);


	div_sum_kernel_3<Dtype><<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim/4/4,inv_var_I.gpu_data(),cov_Ip.gpu_data(),a.mutable_gpu_data());//a = cov_Ip ./ inv_var_I;

	box_filter_gpu(num,channels*maxStates,height/4,width/4,radius,a.gpu_data(),mean_a.mutable_gpu_data(),buffer_image_score.mutable_gpu_data());

	substract_vector_matrix_product_kernel_3<Dtype><<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
  (num,channels,maxStates,spatial_dim/4/4,mean_p.gpu_data(),a.gpu_data(),mean_I.gpu_data(),b.mutable_gpu_data());//	b = mean_p - mean_I .* a;


	box_filter_gpu(num,maxStates,height/4,width/4,radius,b.gpu_data(),mean_b.mutable_gpu_data(),buffer_score.mutable_gpu_data());
	
	//upsample 
	//mean_a mean_a_up
	//mean_b mean_b_up
	interp<Dtype><<<CAFFE_GET_BLOCKS(num*channels*maxStates*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels*maxStates,height/4,width/4,height,width,mean_a.gpu_data(),mean_a_up.mutable_gpu_data());

	interp<Dtype><<<CAFFE_GET_BLOCKS(num*maxStates*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num,maxStates,height/4,width/4,height,width,mean_b.gpu_data(),mean_b_up.mutable_gpu_data());
	
	vector_matrix_product_sum_kernel_3<Dtype><<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim,mean_a_up.gpu_data(),I,mean_b_up.gpu_data(),output_p);// q = I .* mean_a + mean_b;
	
}

template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
	const Dtype * nodePot = bottom[0]->gpu_data();


	int num = bottom[0]->num();
	int maxStates = bottom[0]->channels();
	int channels = bottom[1]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int spatial_dim=height*width;

	int nNodes = num*width *height;
	
	subsampling<Dtype><<<CAFFE_GET_BLOCKS(num*channels*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
  	(num,channels,height,width,height/4,width/4,bottom[1]->gpu_data(),I_sub.mutable_gpu_data());
  	
	//******************************** image ************************************
	box_filter_gpu(num,channels,height/4,width/4,radius,I_sub.gpu_data(),mean_I.mutable_gpu_data(),buffer_image.mutable_gpu_data());

	vector_product_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*channels*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,channels,spatial_dim/4/4,I_sub.gpu_data(),I_sub.gpu_data(),II.mutable_gpu_data());// II = I .* I;

	box_filter_gpu(num,channels*channels,height/4,width/4,radius,II.gpu_data(),mean_II.mutable_gpu_data(),buffer_image_image.mutable_gpu_data());

	substract_vector_product_kernel<Dtype><<<CAFFE_GET_BLOCKS(num*channels*channels*spatial_dim/4/4), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,channels,spatial_dim/4/4,mean_II.gpu_data(),mean_I.gpu_data(),mean_I.gpu_data(), var_I.mutable_gpu_data());//var_I = mean_II - mean_I .* mean_I;



	for(int iter = 0; iter < maxIter; iter++)
	{
		if(iter == 0)
			softmax_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(nNodes), CAFFE_CUDA_NUM_THREADS>>>
			(maxStates,nNodes,nodePot           ,nodeBel[0]->mutable_gpu_data());
		else
			softmax_forward_kernel<Dtype><<<CAFFE_GET_BLOCKS(nNodes), CAFFE_CUDA_NUM_THREADS>>>
			(maxStates,nNodes,tempPot.gpu_data(),nodeBel[iter]->mutable_gpu_data());


		guided_filter_gpu(num,channels,maxStates,height,width,bottom[1]->gpu_data(),I_sub.gpu_data(),nodeBel[iter]->gpu_data(),temp_sub.mutable_gpu_data(),filterPot.mutable_gpu_data());
	

		caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, maxStates, nNodes, maxStates,
													(Dtype)1., this->blobs_[0]->gpu_data(), filterPot.gpu_data(),
													(Dtype)0., compatPot.mutable_gpu_data());

		caffe_gpu_add(maxStates*nNodes,Dtype(1),nodePot,alpha,compatPot.gpu_data(),tempPot.mutable_gpu_data());
	}
	caffe_copy(top[0]->count(),tempPot.gpu_data(),top[0]->mutable_gpu_data());
}
template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom)
{
}
template <typename Dtype>
void FastGuidedCRFLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
}
INSTANTIATE_LAYER_GPU_FUNCS(FastGuidedCRFLayer);
}  // namespace caffe
