// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.
template <typename Dtype>
class Filler 
{
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) 
  {}
  virtual ~Filler() 
  {}
  virtual void Fill(Blob<Dtype>* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler

template <typename Dtype>
class BinaryFiller : public Filler<Dtype> 
{
 public:
  explicit BinaryFiller(const FillerParameter& param) : Filler<Dtype>(param) 
  {}
  virtual void Fill(Blob<Dtype>* blob) 
  {
    CHECK(blob->count());
    //for (int i=0;i<blob->count();i++)
    //{
    //	int state = caffe_rng_rand()%3-1;
    //	blob->mutable_cpu_data()[i] = Dtype(state);
    //}
    caffe_gpu_set(blob->count(),Dtype(0),blob->mutable_gpu_data());
    int num = blob->num();
    int channels = blob->channels();
    int height = blob->height();
    int width = blob->width();
    for (int n=0;n<num;n++)
    	for (int h=0;h<height;h++)
    		for (int w=0;w<width;w++)
    		{
    			int c = n;
    			blob->mutable_cpu_data()[((n*channels+c)*height+h)*width+w] = Dtype(caffe_rng_rand()%3-1);
    		}
  }
};

template <typename Dtype>
class MSRAFiller : public Filler<Dtype> 
{
 public:
  explicit MSRAFiller(const FillerParameter& param) : Filler<Dtype>(param) 
  {}
  virtual void Fill(Blob<Dtype>* blob) 
  {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
   
    Dtype std = sqrt(Dtype(4) / Dtype(fan_in + fan_out));
   // caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std, blob->mutable_cpu_data());
   caffe_rng_uniform<Dtype>(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
   
  }
};

template <typename Dtype>
class GlorotFiller : public Filler<Dtype> 
{
 public:
  explicit GlorotFiller(const FillerParameter& param) : Filler<Dtype>(param) 
  {}
  virtual void Fill(Blob<Dtype>* blob) 
  {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    Dtype std = sqrt(Dtype(2) / Dtype(fan_in + fan_out));
   // caffe_rng_gaussian<Dtype>(blob->count(), Dtype(0), std, blob->mutable_cpu_data());
   caffe_rng_uniform<Dtype>(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
  }
};

template <typename Dtype>
class GaussianFiller : public Filler<Dtype> {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler<Dtype>(param) {}
  virtual void Fill(Blob<Dtype>* blob) 
  {
    Dtype* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    //caffe_rng_gaussian<Dtype>(blob->count(), Dtype(this->filler_param_.mean()), Dtype(this->filler_param_.std()), blob->mutable_cpu_data());
    Dtype std = this->filler_param_.std();
    caffe_rng_uniform<Dtype>(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
  }
};

template <typename Dtype>
class BilinearFiller : public Filler<Dtype> 
{
 public:
  explicit BilinearFiller(const FillerParameter& param) : Filler<Dtype>(param) 
  {}
  virtual void Fill(Blob<Dtype>* blob) {
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    Dtype* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) 
    {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
  }
};

template <typename Dtype>
class PottsFiller : public Filler<Dtype> 
{
 public:
  explicit PottsFiller(const FillerParameter& param): Filler<Dtype>(param) 
  {}
  virtual void Fill(Blob<Dtype>* blob) 
  {
    Dtype* data = blob->mutable_cpu_data();
    caffe_set(blob->count(),Dtype(0),data);
   	for(int i=0;i<blob->num();i++)
   		data[i+i*blob->channels()]=1;
  }
};

template <typename Dtype>
Filler<Dtype>* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type(); 
  if (type == "binary") 
    return new BinaryFiller<Dtype>(param);
  else if (type == "msra") 
    return new MSRAFiller<Dtype>(param);
  else if (type == "bilinear") 
    return new BilinearFiller<Dtype>(param);
  else if (type == "potts") 
    return new PottsFiller<Dtype>(param);  
  else if (type == "gaussian") 
    return new GaussianFiller<Dtype>(param);
  else if (type == "glorot") 
    return new GlorotFiller<Dtype>(param);
  else 
    CHECK(false) << "Unknown filler name: " << param.type();
  return (Filler<Dtype>*)(NULL);
}

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
