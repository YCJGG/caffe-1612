/*
* Integrate imgResize and blobAlign python layers together.
*
* Author: Wei Zhen @ IIE, CAS
* Last modified: 2017-06-11
*/

#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/resize_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void nearestForwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	const Dtype* bottom_data, Dtype* top_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int rw = index % output_size;
		int rh = (index / output_size) % output_size;
		int c = (index / output_size / output_size) % input_channels;
		int n = index / output_size / output_size / input_channels;
		// indexing and sampling
		int h = int(rh / resize_factor);
		int w = int(rw / resize_factor);
		h = min(h, input_size);
		w = min(w, input_size);
		bottom_data += (n * input_channels + c) * input_size * input_size;
		top_data[index] = bottom_data[h * input_size + w];
	}
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();

  // perform interpolation based on different interpolation types
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case ResizeParameter_InterpolationType_NEAREST:
		// parallel at pixel level
		nearestForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
			(count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_data, top_data);
		break;
	case ResizeParameter_InterpolationType_BILINEAR:
		// not implemented
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }
}

template <typename Dtype>
__global__ void nearestBackwardGPU(const int nthreads,
	const int input_channels, const int input_size, const int output_size, const float resize_factor,
	Dtype* bottom_diff, const Dtype* top_diff) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		// resume channel idx and pixel idx
		int w = index % input_size;
		int h = (index / input_size) % input_size;
		int c = (index / input_size / input_size) % input_channels;
		int n = index / input_size / input_size / input_channels;
		// indexing and sampling
		int hstart = int(h * resize_factor);
		int wstart = int(w * resize_factor);
		int hend = int(hstart + resize_factor);
		int wend = int(wstart + resize_factor);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, output_size);
		wend = min(wend, output_size);
		top_diff += (n*input_channels + c) * output_size * output_size;
		for (int rh = hstart; rh < hend; ++rh) {
		    for (int rw = wstart; rw < wend; ++rw) {
			bottom_diff[index] += top_diff[rh * output_size + rw];
		    }
		}
		bottom_diff[index] /= (hend-hstart) * (wend-wstart);
	}
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {    return;  }

  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);

  // data gradient
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case ResizeParameter_InterpolationType_NEAREST:
		nearestBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
			(count, bottom[0]->channels(), this->input_size_, this->output_size_,
			 this->resize_factor_, bottom_diff, top_diff);
		break;
	case ResizeParameter_InterpolationType_BILINEAR:
		// not implemented
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }

}

INSTANTIATE_LAYER_GPU_FUNCS(ResizeLayer);

}// namespace caffe
