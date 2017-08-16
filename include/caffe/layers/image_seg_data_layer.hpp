#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "boost/scoped_ptr.hpp"
#include "hdf5.h"
#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/blocking_queue.hpp"
#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
class ImageDimPrefetchingDataLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit ImageDimPrefetchingDataLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageDimPrefetchingDataLayer() {}
  // LayerSetUp: implements common data layer setup functionality, and calls
  // DataLayerSetUp to do special data layer setup for individual layer types.
  // This method may not be overridden.
  void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  //virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
   //   const vector<Blob<Dtype>*>& top);

  // The thread's function
  virtual void InternalThreadEntry() {}

 protected:
  Blob<Dtype> prefetch_data_dim_;
  bool output_data_dim_;
};


template <typename Dtype>
class ImageSegDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
 public:
  explicit ImageSegDataLayer(const LayerParameter& param)
    : ImageDimPrefetchingDataLayer<Dtype>(param) {}
  virtual ~ImageSegDataLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "ImageSegData"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 3; }
  virtual inline bool AutoTopBlobs() const { return true; }

 protected:
  virtual void ShuffleImages();
  virtual void InternalThreadEntry();
  virtual inline void load_batch(Batch<Dtype>* batch) {return ;}

 protected:
  Blob<Dtype> transformed_label_;

  shared_ptr<Caffe::RNG> prefetch_rng_;

  vector<std::pair<std::string, std::string> > lines_;
  int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
