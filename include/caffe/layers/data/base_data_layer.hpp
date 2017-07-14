#ifndef CAFFE_BASE_DATA_LAYERS_HPP_
#define CAFFE_BASE_DATA_LAYERS_HPP_
#include <boost/thread.hpp>

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"




namespace caffe {

template <typename Dtype>
class BaseDataLayer : public Layer<Dtype>
{
 public:
  explicit BaseDataLayer(const LayerParameter& param): Layer<Dtype>(param), transform_param_(param.transform_param()), thread_() {}
  virtual ~BaseDataLayer();
  
 	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom);
  virtual void SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {}
	virtual void InternalThreadEntry(int gpu_id) {}
 protected:
 
  TransformationParameter transform_param_;
  shared_ptr<DataTransformer<Dtype> > data_transformer_;


	Blob<Dtype> prefetch_data_;
  Blob<Dtype> prefetch_label_;
  
  Blob<Dtype> transformed_data_;
  Blob<Dtype> transformed_label_;
  
  //int data_iter_;
  int gpu_id_;
  shared_ptr<boost::thread> thread_;

};

}  // namespace caffe

#endif  // CAFFE_DATA_LAYERS_HPP_
