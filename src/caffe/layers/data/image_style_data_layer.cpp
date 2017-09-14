#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

#include "caffe/layers/data/image_style_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/solver.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

template <typename Dtype>
void ImageStyleDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
 
}
// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageStyleDataLayer<Dtype>::InternalThreadEntry(int gpu_id)
{
 
}

template <typename Dtype>
ImageStyleDataLayer<Dtype>::~ImageStyleDataLayer<Dtype>() 
{
 
}

INSTANTIATE_CLASS(ImageStyleDataLayer);
REGISTER_LAYER_CLASS(ImageStyleData);
}  // namespace caffe
