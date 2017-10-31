/*
# Give weights on gradient.
#
# Author: Wei Zhen @ IIE, CAS
# Create on: 2016-09-11
# Last modify: 2016-09-11
#
*/

#ifndef CAFFE_WEIGHTEDGRADIENT_LAYER_HPP_
#define CAFFE_WEIGHTEDGRADIENT_LAYER_HPP_

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include <vector>
#include <algorithm>

namespace caffe {

template <typename Dtype>
class WeightedGradientLayer: public Layer<Dtype> {
 public:
  explicit WeightedGradientLayer(const LayerParameter& param)
      : Layer<Dtype>(param), weight(1) {};

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "WeightedGradient"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }


 protected:
  /**
   * @param bottom input Blob vector (length 2)
   * @param top output Blob vector (length 1)
   */
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  /**
   * @brief Computes the error gradient w.r.t. the Inflation layer input and the factor.
   *
   * @param top output Blob vector (length 1), providing the error gradient with
   *      respect to the outputs
   * @param propagate_down is unuseful in this implementation
   * @param bottom input Blob vector (length 2)
   */
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  vector<int> label_set;
  float weight;
};

}  // namespace caffe

#endif  // CAFFE_WIEGHTEDGRADIENT_LAYER_HPP_
