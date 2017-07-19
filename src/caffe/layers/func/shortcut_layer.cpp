#include <vector> 

#include "caffe/layers/func/shortcut_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


template <typename Dtype>
void ShortcutLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
 
}	

template <typename Dtype>
void ShortcutLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
  if (bottom[0]->height() != bottom[1]->height() )
    LOG(FATAL)<<"wrong size";
  
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(),bottom[1]->height());
  CHECK_EQ(bottom[0]->width(),bottom[1]->width());
  
  top[0]->ReshapeLike(*bottom[0]);
}
template <typename Dtype>
ShortcutLayer<Dtype>::~ShortcutLayer()
{
}


INSTANTIATE_CLASS(ShortcutLayer);
REGISTER_LAYER_CLASS(Shortcut);
}  // namespace caffe
