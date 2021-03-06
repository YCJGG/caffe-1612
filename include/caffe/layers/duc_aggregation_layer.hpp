/*
# DUC aggregation operation.
# A re-implementation of DUC operation described in paper "Understanding Convolution for Semantic Segmentation".
#
# Author: Wei Zhen @ CS, HUST
# Create on: 2017-08-17
# Last modify: 2017-08-17
#
*/

#ifndef CAFFE_DUC_AGGREGATION_LAYER_HPP_
#define CAFFE_DUC_AGGREGATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"

namespace caffe {

template <typename Dtype>
class DUCAggregationLayer : public Layer<Dtype> {
 public:
  explicit DUCAggregationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "DUCAggregation"; }

  virtual inline int ExactNumBottomBlobs() const { return 1; }

  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);


  int upsampling_factor;
};

}  // namespace caffe

#endif  // CAFFE_DUC_AGGREGATION_LAYER_HPP_
