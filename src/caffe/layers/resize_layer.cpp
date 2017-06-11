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
void ResizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // do nothing
}

template <typename Dtype>
void ResizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // parse params and get resize factor
  this->input_size_ = bottom[0]->height();

  ResizeParameter resize_param = this->layer_param_.resize_param();
  switch (resize_param.function_type()) {
	case ResizeParameter_FunctionType_SIZE_GIVEN:
		if (resize_param.output_size() <= 0) {
		    LOG(INFO)<< "Illegal output size (" << resize_param.output_size() << "), use default resize factor 1.";
		}
		else {
		    this->output_size_ = resize_param.output_size();
		    this->resize_factor_ = (float)this->output_size_ / this->input_size_;
		}
		break;
	case ResizeParameter_FunctionType_BLOB_ALIGN:
		CHECK_EQ(bottom.size(), 2) << "When using BLOB_ALIGN option, bottom[1] must be given.";
		this->output_size_ = bottom[1]->height();
		this->resize_factor_ = (float)this->output_size_ / this->input_size_;
		break;
	case ResizeParameter_FunctionType_FACTOR_GIVEN:
		CHECK_GT(resize_param.resize_factor(), 0) << "Illegal resize factor (" << resize_param.resize_factor() << ").";
		this->resize_factor_ = resize_param.resize_factor();
		this->output_size_ = static_cast<int>(this->input_size_ * this->resize_factor_);
		break;
	default:
		this->resize_factor_ = 1;
		this->output_size_ = this->input_size_;
  }

  // reshape top (assume that all features are square)
  top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), this->output_size_, this->output_size_);
}

template <typename Dtype>
void ResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // perform interpolation based on different interpolation types
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case ResizeParameter_InterpolationType_NEAREST:
		for (int n = 0; n < bottom[0]->num(); ++n) {
		    for (int c = 0; c < bottom[0]->channels(); ++c) {
			for (int rh = 0; rh < this->output_size_; ++rh) {
			    for (int rw = 0; rw < this->output_size_; ++rw) {
				int h = int(rh / this->resize_factor_);
				int w = int(rw / this->resize_factor_);
				h = std::min(h, this->input_size_);
				w = std::min(w, this->input_size_);
				top_data[rh * this->output_size_ + rw] = bottom_data[h * this->input_size_ + w];
			    }
			}
			// compute offset
			bottom_data += bottom[0]->offset(0, 1);
			top_data += top[0]->offset(0, 1);
		    }
		}
		break;
	case ResizeParameter_InterpolationType_BILINEAR:
		// not implemented
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }
}

template <typename Dtype>
void ResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {    return;  }

  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);

  // data gradient
  switch (this->layer_param_.resize_param().intepolation_type()) {
	case ResizeParameter_InterpolationType_NEAREST:
		for (int n = 0; n < top[0]->num(); ++n) {
		    for (int c = 0; c < top[0]->channels(); ++c) {
			for (int h = 0; h < this->input_size_; ++h) {
			    for (int w = 0; w < this->input_size_; ++w) {
				int hstart = int(h * this->resize_factor_);
				int wstart = int(w * this->resize_factor_);
				int hend = int(hstart + this->resize_factor_);
				int wend = int(wstart + this->resize_factor_);
				hstart = std::max(hstart, 0);
				wstart = std::max(wstart, 0);
				hend = std::min(hend, this->output_size_);
				wend = std::min(wend, this->output_size_);
				for (int rh = hstart; rh < hend; ++rh) {
				    for (int rw = wstart; rw < wend; ++rw) {
					bottom_diff[h * this->input_size_ + w] += top_diff[rh * this->output_size_ + rw];
				    }
				}
				bottom_diff[h * this->input_size_ + w] /= (hend-hstart) * (wend-wstart);
			    }
			}
			// offset
			bottom_diff += bottom[0]->offset(0, 1);
			top_diff += top[0]->offset(0, 1);
		    }
		}
		break;
	case ResizeParameter_InterpolationType_BILINEAR:
		// not implemented
		break;
	default:
		LOG(FATAL) << "Unknown interpolation type.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(ResizeLayer);
#endif

INSTANTIATE_CLASS(ResizeLayer);
REGISTER_LAYER_CLASS(Resize);
}  // namespace caffe
