#ifndef CAFFE_IMAGE_Class_DATA_LAYERS_HPP_
#define CAFFE_IMAGE_Class_DATA_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"

#include "caffe/layers/data/base_data_layer.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/net.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"

namespace caffe {

template <typename Dtype>
class ImageClassDataLayer : public BaseDataLayer<Dtype> 
{
 public:
  explicit ImageClassDataLayer(const LayerParameter& param) : BaseDataLayer<Dtype>(param) {}
  virtual ~ImageClassDataLayer();
  virtual inline const char* type() const { return "ImageClassData"; }
  
  
  
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
	virtual void InternalThreadEntry(int gpu_id);
 protected:


  vector<vector<std::pair<std::string, std::string> > > lines_;
  vector<int> lines_id_;
  int cur_class;

};


}  // namespace caffe

#endif  // CAFFE_BASE_DATA_LAYERS_HPP_
