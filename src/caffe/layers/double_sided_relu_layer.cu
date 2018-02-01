#include <algorithm>
#include <vector>

#include "caffe/layers/double_sided_relu_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void DoubleReLUForward(const int n, const Dtype* in, Dtype* out,
    Dtype max_value, Dtype min_value) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = (in[index] > min_value) ? in[index] : min_value;
    out[index] = (out[index] < max_value) ? out[index] : max_value;
  }
}

template <typename Dtype>
void DoubleSidedReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  Dtype max_value = this->layer_param_.double_sided_relu_param().max_value();
  Dtype min_value = this->layer_param_.double_sided_relu_param().min_value();
  DoubleReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, max_value, min_value);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void DoubleReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff, Dtype max_value, Dtype min_value) {
  CUDA_KERNEL_LOOP(index, n) {
    out_diff[index] = in_diff[index] * ((in_data[index] > min_value) && (in_data[index] < max_value));
  }
}

template <typename Dtype>
void DoubleSidedReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int count = bottom[0]->count();
    Dtype max_value = this->layer_param_.double_sided_relu_param().max_value();
    Dtype min_value = this->layer_param_.double_sided_relu_param().min_value();
    DoubleReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff, max_value, min_value);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(DoubleSidedReLULayer);


}  // namespace caffe
