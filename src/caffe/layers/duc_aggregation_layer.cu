/*
# DUC aggregation operation.
#
# Author: Wei Zhen
# Create on: 2017-08-21
# Last modify: 2017-08-21
#
*/

#include "caffe/layers/duc_aggregation_layer.hpp"

#define MAX_UPSAMPLING_FACTOR 8

namespace caffe {

template <typename Dtype>
__global__ void DUC_GPUForward(const int nthreads,
		const Dtype* bottom_data, Dtype* top_data, int upsampling_factor, int sqrt_up_f, 
		int top_channels, int top_height, int top_width,
		int bottom_channels, int bottom_height, int bottom_width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
	// resume pix indices
	int in_w = index % bottom_width;
	int in_h = (index / bottom_width) % bottom_height;
	int bottom_inner_count = bottom_width * bottom_height;
	int out_c = (index / bottom_inner_count) % top_channels;
	int n = index / bottom_inner_count / top_channels;
	// current data offset
	top_data += ((n * top_channels) + out_c) * top_height * top_width;
	bottom_data += ((n * bottom_channels) + out_c * sqrt_up_f) * bottom_inner_count;
	// iterating pix on top feature map (out_) and fetch correspond feature in bottom blob (in_)
	for (int up_idx = 0; up_idx < sqrt_up_f; up_idx++) {
	    int out_h = in_h*upsampling_factor + int(up_idx/upsampling_factor);
	    int out_w = in_w*upsampling_factor + up_idx%upsampling_factor;
	    int in_c = up_idx;
	    top_data[out_h * top_width + out_w] = bottom_data[((in_c * bottom_height) + in_h) * bottom_width + in_w];
	}
    }
}

template <typename Dtype>
void DUCAggregationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // get top and bottom data
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();

    int sqrt_up_f = this->upsampling_factor*this->upsampling_factor;
    int count = top[0]->num() * top[0]->channels() * bottom[0]->height() * bottom[0]->width();
    DUC_GPUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
					bottom_data, top_data, this->upsampling_factor, sqrt_up_f,
					top[0]->channels(), top[0]->height(), top[0]->width(),
					bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
__global__ void DUC_GPUBackward(const int nthreads,
		Dtype* bottom_diff, const Dtype* top_diff, int upsampling_factor, int sqrt_up_f, 
		int top_channels, int top_height, int top_width,
		int bottom_channels, int bottom_height, int bottom_width) {
    CUDA_KERNEL_LOOP(index, nthreads) {
	// resume pix indices
	int in_w = index % bottom_width;
	int in_h = (index / bottom_width) % bottom_height;
	int bottom_inner_count = bottom_width * bottom_height;
	int out_c = (index / bottom_inner_count) % top_channels;
	int n = index / bottom_inner_count / top_channels;
	// current data offset
	top_diff += ((n * top_channels) + out_c) * top_height * top_width;
	bottom_diff += ((n * bottom_channels) + out_c * sqrt_up_f) * bottom_inner_count;
	// iterating pix on top feature map (out_) and fetch correspond feature in bottom blob (in_)
	for (int up_idx = 0; up_idx < sqrt_up_f; up_idx++) {
	    int out_h = in_h*upsampling_factor + int(up_idx/upsampling_factor);
	    int out_w = in_w*upsampling_factor + up_idx%upsampling_factor;
	    int in_c = up_idx;
	    bottom_diff[((in_c * bottom_height) + in_h) * bottom_width + in_w] = top_diff[out_h * top_width + out_w];
	}
    }
}

template <typename Dtype>
void DUCAggregationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // get top and bottom diff
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();

    int sqrt_up_f = this->upsampling_factor*this->upsampling_factor;
    int count = top[0]->num() * top[0]->channels() * bottom[0]->height() * bottom[0]->width();
    DUC_GPUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count,
					bottom_diff, top_diff, this->upsampling_factor, sqrt_up_f,
					top[0]->channels(), top[0]->height(), top[0]->width(),
					bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

INSTANTIATE_LAYER_GPU_FUNCS(DUCAggregationLayer);
}  // namespace caffe
