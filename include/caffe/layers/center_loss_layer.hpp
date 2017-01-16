#ifndef CAFFE_CENTER_LOSS_LAYER_HPP_
#define CAFFE_CENTER_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CenterLossLayer : public LossLayer<Dtype> {
 public:
  explicit CenterLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param) {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterLoss"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return -1; }
  virtual inline int MinTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  // convert index on bottom[0] (data) onto bottom[1] (label)
  inline int label_idx_converter(int num, int label_height, int label_width, int data_idx, int data_width);

  float label_bottom_factor;
  int label_axis_, outer_num_, inner_num_, label_num_;
  // set of ignore labels
  vector<int> ignore_label_;
  
  Blob<int> label_counter_;
  Blob<Dtype> distance_;
  Blob<Dtype> variation_sum_;

  // center inter distances
  Blob<Dtype> center_mutual_distance;

  // iter counter
  int count_;
};

}  // namespace caffe

#endif  // CAFFE_CENTER_LOSS_LAYER_HPP_
