#include <vector>

#include "caffe/layers/loss/parse_evaluate_layer.hpp"

namespace caffe {

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
	  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  const Dtype* bottom_pred = bottom[0]->cpu_data();
  const Dtype* bottom_gt = bottom[1]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  caffe_set(top[0]->count(), Dtype(0), top_data);
  int num = bottom[0]->num();
  int spatial_dim = bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < num; ++i) {
    // count the number of ground truth labels, the predicted labels, and
    // predicted labels happens to be ground truth labels
    for (int j = 0; j < spatial_dim; ++j) {
      int gt_label = bottom_gt[j];
      int pred_label = bottom_pred[j];
      CHECK_LT(pred_label, num_labels_);
      if (ignore_labels_.find(gt_label) != ignore_labels_.end()) {
        continue;
      }
      if (gt_label == pred_label) {
        top_data[gt_label * 3]++;
      }
      top_data[gt_label * 3 + 1]++;
      top_data[pred_label * 3 + 2]++;
    }
    bottom_pred += bottom[0]->offset(1);
    bottom_gt += bottom[1]->offset(1);
  }
  // ParseEvaluate layer should not be used as a loss function.
	//for (int i=0;i<bottom[0]->count();i++)
	//{
	//	if (bottom[0]->cpu_data()[i] != bottom[1]->cpu_data()[i])
	//	{
			//LOG(INFO)<< i / bottom[0]->width()<<", "<<i % bottom[0]->width();
	//		LOG(INFO)<<bottom[0]->cpu_data()[i]<<", "<<bottom[1]->cpu_data()[i];
	//	}
	//}
	//LOG(FATAL)<<"-------------";
}

template <typename Dtype>
void ParseEvaluateLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<Blob<Dtype>*>& bottom) 
{
}
template <typename Dtype>
void ParseEvaluateLayer<Dtype>::SecForward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) 
{
}
INSTANTIATE_LAYER_GPU_FUNCS(ParseEvaluateLayer);

}  // namespace caffe
