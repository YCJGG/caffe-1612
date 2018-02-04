#include <algorithm>
#include <vector>

#include "caffe/layers/double_sided_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void DoubleSidedReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  Dtype max_value = this->layer_param_.double_sided_relu_param().max_value();
  Dtype min_value = this->layer_param_.double_sided_relu_param().min_value();
  for (int i = 0; i < count; ++i) {
    top_data[i] = std::min(std::max(bottom_data[i], min_value), max_value);
  }
}

template <typename Dtype>
void DoubleSidedReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype max_value = this->layer_param_.double_sided_relu_param().max_value();
    Dtype min_value = this->layer_param_.double_sided_relu_param().min_value();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * ((bottom_data[i] > min_value) && (bottom_data[i] < max_value));
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(DoubleSidedReLULayer);
#endif

INSTANTIATE_CLASS(DoubleSidedReLULayer);
REGISTER_LAYER_CLASS(DoubleSidedReLU);

}  // namespace caffe
