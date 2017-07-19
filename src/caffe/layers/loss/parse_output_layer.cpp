#include <vector>

#include "caffe/layers/loss/parse_output_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParseOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  out_max_val_ = top.size() > 1;
}

template <typename Dtype>
void ParseOutputLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Produces max_ind and max_val
  top[0]->Reshape(bottom[0]->num(), 1, bottom[0]->height(), bottom[0]->width());
  if (out_max_val_) {
    top[1]->ReshapeLike(*top[0]);
  }
  max_prob_.Reshape(1, 1, bottom[0]->height(), bottom[0]->width());
}



INSTANTIATE_CLASS(ParseOutputLayer);
REGISTER_LAYER_CLASS(ParseOutput);

}  // namespace caffe
