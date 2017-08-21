/*
# DUC aggregation operation.
#
# Author: Wei Zhen
# Create on: 2017-08-17
# Last modify: 2017-08-17
#
*/

#include "caffe/layers/duc_aggregation_layer.hpp"

#define MAX_UPSAMPLING_FACTOR 8

namespace caffe {

template <typename Dtype>
void DUCAggregationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // get and check duc upsampling factor
    DUCParameter duc_param = this->layer_param().duc_param();
    this->upsampling_factor = duc_param.upsampling_factor();
    // check value
    CHECK_LE(this->upsampling_factor, MAX_UPSAMPLING_FACTOR) << "The DUC upsampling factor should be less than MAX_UPSAMPLING_FACTOR: ("
							     << this->upsampling_factor << " vs. "<< MAX_UPSAMPLING_FACTOR << ").";
    CHECK_GE(this->upsampling_factor, 1) << "The DUC upsampling factor should be greater than 1.";
    // check input blob channel
    CHECK_EQ(bottom[0]->channels() % (this->upsampling_factor*this->upsampling_factor), 0)
							     << "The input channel number should be INT times (upsampling_factor)^2: ("
							     << bottom[0]->channels() << " vs. "
							     << this->upsampling_factor << "*" << this->upsampling_factor << ").";
}

template <typename Dtype>
void DUCAggregationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // get output blob size and reshape top blob
    int top_channel = bottom[0]->channels() / this->upsampling_factor / this->upsampling_factor;
    int top_height = bottom[0]->height() * this->upsampling_factor;
    int top_width = bottom[0]->width() * this->upsampling_factor;
    top[0]->Reshape(bottom[0]->num(), top_channel, top_height, top_width);
}

template <typename Dtype>
void DUCAggregationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // get top and bottom data
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();

    int top_width = top[0]->width();
    int bottom_height = bottom[0]->height();
    int bottom_width = bottom[0]->width();
    int sqrt_up_f = this->upsampling_factor*this->upsampling_factor;
    // iterating pix on top feature map (out_) and fetch correspond feature in bottom blob (in_)
    for (int n = 0; n < top[0]->num(); n++) {
        for (int out_c = 0; out_c < top[0]->channels(); out_c++) {
	    for (int in_h = 0; in_h < bottom[0]->height(); in_h++) {
		for (int in_w = 0; in_w < bottom_width; in_w++) {
		    for (int up_idx = 0; up_idx < sqrt_up_f; up_idx++) {
			int out_h = in_h*this->upsampling_factor + int(up_idx/this->upsampling_factor);
			int out_w = in_w*this->upsampling_factor + up_idx%this->upsampling_factor;
			int in_c = up_idx;
			top_data[out_h * top_width + out_w] = bottom_data[((in_c * bottom_height) + in_h) * bottom_width + in_w];
		    }
		}
	    }
	    bottom_data += bottom[0]->offset(0, sqrt_up_f);
	    top_data += top[0]->offset(0, 1);
        }
    }
}

template <typename Dtype>
void DUCAggregationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    // get top and bottom diff
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();

    int top_width = top[0]->width();
    int bottom_height = bottom[0]->height();
    int bottom_width = bottom[0]->width();
    int sqrt_up_f = this->upsampling_factor*this->upsampling_factor;
    // iterating on top gradient map (out_) and give correspond diff to bottom blob (in_)
    for (int n = 0; n < top[0]->num(); n++) {
        for (int out_c = 0; out_c < top[0]->channels(); out_c++) {
	    for (int in_h = 0; in_h < bottom_height; in_h++) {
		for (int in_w = 0; in_w < bottom_width; in_w++) {
		    for (int up_idx = 0; up_idx < sqrt_up_f; up_idx++) {
			int out_h = in_h*this->upsampling_factor + int(up_idx/this->upsampling_factor);
			int out_w = in_w*this->upsampling_factor + up_idx%this->upsampling_factor;
			int in_c = up_idx;
			bottom_diff[((in_c * bottom_height) + in_h) * bottom_width + in_w] = top_diff[out_h * top_width + out_w];
		    }
		}
	    }
	    bottom_diff += bottom[0]->offset(0, sqrt_up_f);
	    top_diff += top[0]->offset(0, 1);
        }
    }
}

#ifdef CPU_ONLY
STUB_GPU(DUCAggregationLayer);
#endif
INSTANTIATE_CLASS(DUCAggregationLayer);
REGISTER_LAYER_CLASS(DUCAggregation);
}  // namespace caffe
