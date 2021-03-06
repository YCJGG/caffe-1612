#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_DENSE_P_NORM)
        << "Padding implemented only for average, max and p_norm pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->height();
    kernel_w_ = bottom[0]->width();
  }
  pooled_height_ = static_cast<int>(ceil(static_cast<float>(
      height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
  pooled_width_ = static_cast<int>(ceil(static_cast<float>(
      width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
        pooled_width_);
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(bottom[0]->num(), channels_, pooled_height_,
      pooled_width_);
  }
  // add on 2018-01-15
  // If dense p-norm pooling, bottom[1] should have the same size as
  // the output blob
  if (this->layer_param_.pooling_param().pool() ==
    PoolingParameter_PoolMethod_DENSE_P_NORM) {
      CHECK_EQ(bottom[1]->channels(), top[0]->channels())
	<< "P values should have the same number of channels as output, ("
	<< bottom[1]->channels() << " vs. " << top[0]->channels() << ")";
      CHECK_EQ(bottom[1]->height(), top[0]->height())
	<< "P values should have the same height as output, ("
	<< bottom[1]->height() << " vs. " << top[0]->height() << ")";
      CHECK_EQ(bottom[1]->width(), top[0]->width())
	<< "P values should have the same width as output, ("
	<< bottom[1]->width() << " vs. " << top[0]->width() << ")";
    // resize numerator and denominator to save internal results and avoid 
    // multiple computations
    numerator.Reshape(top[0]->num(),top[0]->channels(),top[0]->height(),top[0]->width());
    denominator.Reshape(top[0]->num(),top[0]->channels(),top[0]->height(),top[0]->width());
    padded_height_ = height_ + 2*pad_h_;
    padded_width_ = width_ + 2*pad_w_;
    padded_bottom.Reshape(bottom[0]->num(), channels_, padded_height_, padded_width_);
  }
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitalized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }
    caffe_set(top_count, Dtype(-FLT_MAX), top_data);
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_);
            int wend = min(wstart + kernel_w_, width_);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            const int pool_index = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                const int index = h * width_ + w;
                if (bottom_data[index] > top_data[pool_index]) {
                  top_data[pool_index] = bottom_data[index];
                  if (use_top_mask) {
                    top_mask[pool_index] = static_cast<Dtype>(index);
                  } else {
                    mask[pool_index] = index;
                  }
                }
              }
            }
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    for (int i = 0; i < top_count; ++i) {
      top_data[i] = 0;
    }
    // The main loop
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                top_data[ph * pooled_width_ + pw] +=
                    bottom_data[h * width_ + w];
              }
            }
            top_data[ph * pooled_width_ + pw] /= pool_size;
          }
        }
        // compute offset
        bottom_data += bottom[0]->offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  // add on 2018-01-15, dense p_norm pooling
  case PoolingParameter_PoolMethod_DENSE_P_NORM: {
    const Dtype* p_data = bottom[1]->cpu_data();
    // init numerator and denominator
    double* numerator_data = this->numerator.mutable_cpu_data();
    caffe_set(numerator.count(), double(0), numerator_data);
    double* denominator_data = this->denominator.mutable_cpu_data();
    caffe_set(denominator.count(), double(0), denominator_data);
    // bottom[0] padding
    Dtype* padded_bottom_data = padded_bottom.mutable_cpu_data();
    caffe_set(padded_bottom.count(), Dtype(0), padded_bottom_data);
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int h = 0; h < height_; ++h) {
	    caffe_copy(width_, bottom_data+h*width_, padded_bottom_data+(h+pad_h_)*padded_width_+pad_w_);
        }
        bottom_data += bottom[0]->offset(0,1);
        padded_bottom_data += padded_bottom.offset(0,1);
      }
    }
    padded_bottom_data = padded_bottom.mutable_cpu_data();
    // The main loop
    // y = sum(x_i ** (p+1)) / sum(x_i ** p)
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_;
            int wstart = pw * stride_w_;
            int hend = min(hstart + kernel_h_, padded_height_);
            int wend = min(wstart + kernel_w_, padded_width_);
	    double tmp_numerator = 0;
	    double tmp_denominator = Dtype(FLT_MIN);	// avoid divided by 0
	    int top_idx = ph * pooled_width_ + pw;
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
		int bottom_idx = h * padded_width_ + w;
		tmp_numerator += (Dtype)pow(padded_bottom_data[bottom_idx], p_data[top_idx]+1);
		tmp_denominator += (Dtype)pow(padded_bottom_data[bottom_idx], p_data[top_idx]);
              }
            }
	    top_data[top_idx] = tmp_numerator / tmp_denominator;
	    numerator_data[top_idx] = tmp_numerator;
	    denominator_data[top_idx] = tmp_denominator;
          }
        }
        // compute offset
	p_data += bottom[1]->offset(0, 1);
	numerator_data += numerator.offset(0, 1);
	denominator_data += denominator.offset(0, 1);
        padded_bottom_data += padded_bottom.offset(0,1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  }
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            const int index = ph * pooled_width_ + pw;
            const int bottom_index =
                use_top_mask ? top_mask[index] : mask[index];
            bottom_diff[bottom_index] += top_diff[index];
          }
        }
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        if (use_top_mask) {
          top_mask += top[0]->offset(0, 1);
        } else {
          mask += top[0]->offset(0, 1);
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    for (int n = 0; n < top[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_ - pad_h_;
            int wstart = pw * stride_w_ - pad_w_;
            int hend = min(hstart + kernel_h_, height_ + pad_h_);
            int wend = min(wstart + kernel_w_, width_ + pad_w_);
            int pool_size = (hend - hstart) * (wend - wstart);
            hstart = max(hstart, 0);
            wstart = max(wstart, 0);
            hend = min(hend, height_);
            wend = min(wend, width_);
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
                bottom_diff[h * width_ + w] +=
                  top_diff[ph * pooled_width_ + pw] / pool_size;
              }
            }
          }
        }
        // offset
        bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  // add on 2018-01-15, dense p_norm pooling
  case PoolingParameter_PoolMethod_DENSE_P_NORM:  {
    const Dtype* padded_bottom_data = padded_bottom.cpu_data();
    const Dtype* top_data = top[0]->cpu_data();
    Dtype* p_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* p_data = bottom[1]->cpu_data();
    // init numerator and denominator
    const double* numerator_data = this->numerator.cpu_data();
    const double* denominator_data = this->denominator.cpu_data();
    // get denominator**2 in advance
    Blob<double> denominator_pow2;
    denominator_pow2.ReshapeLike(denominator);
    double* denominator_pow2_data = denominator_pow2.mutable_cpu_data();
    caffe_powx(denominator.count(), denominator_data, double(2), denominator_pow2_data);

    // The main loop
    // dL/dx_i = dL/dy_j * [((p_j+1)*(x_i**p_j)*denominator_j) - (p_j*(x_i**(p_j-1))*numerator_j)] / (denominator_j ** 2)
    // dL/dp_j = dL/dy_j * [sum_i(ln(x_i)*(x_i**(p_j+1))*denominator_j - sum_i(ln(x_i)*(x_i**p_j))*numerator_j] / (denominator_j ** 2)
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int c = 0; c < channels_; ++c) {
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            int hstart = ph * stride_h_;
            int wstart = pw * stride_w_;
            int hend = min(hstart + kernel_h_, padded_height_);
            int wend = min(wstart + kernel_w_, padded_width_);
	    int top_idx = ph * pooled_width_ + pw;	// j of p_j, y_j
	    Dtype sum1 = 0;				// sum_i(ln(x_i)*(x_i**(p_j+1))
	    Dtype sum2 = 0;				// sum_i(ln(x_i)*(x_i**(p_j))
            for (int h = hstart; h < hend; ++h) {
              for (int w = wstart; w < wend; ++w) {
		int bottom_idx = h * padded_width_ + w;	// i of x_i
		Dtype x_pow_p_minus1 = (Dtype)pow(padded_bottom_data[bottom_idx], p_data[top_idx]-1);
		Dtype x_pow_p = x_pow_p_minus1 * padded_bottom_data[bottom_idx];
		Dtype x_pow_p_plus1 = x_pow_p * padded_bottom_data[bottom_idx];
		// dL/dx_i
		// directly crop out from padded gradient via idx transform
		if ((h >= pad_h_) && (w >= pad_w_) && (h < height_ + pad_h_) && (w < width_ + pad_w_))
		{
		    bottom_diff[(h-pad_h_) * width_ + (w-pad_w_)] += top_diff[top_idx] * 
		    	( ((p_data[top_idx]+1) * x_pow_p * denominator_data[top_idx])
			 - (p_data[top_idx] * x_pow_p_minus1 * numerator_data[top_idx]) )
		    	/ denominator_pow2_data[top_idx];
		}
		// dL/dp_j
		// avoid x->0 in log()
		Dtype bottom_data_value = padded_bottom_data[bottom_idx];
		sum1 += (Dtype)log(bottom_data_value) * x_pow_p_plus1;
		sum2 += (Dtype)log(bottom_data_value) * x_pow_p;
              }
            }
	    p_diff[top_idx] = top_diff[top_idx] * (sum1*denominator_data[top_idx] - sum2*numerator_data[top_idx]) / denominator_pow2_data[top_idx];
          }
        }
        // compute offset
	p_diff += bottom[1]->offset(0, 1);
	p_data += bottom[1]->offset(0, 1);
	numerator_data += numerator.offset(0, 1);
	denominator_data += denominator.offset(0, 1);
	bottom_diff += bottom[0]->offset(0, 1);
        top_diff += top[0]->offset(0, 1);
        padded_bottom_data += padded_bottom.offset(0, 1);
        top_data += top[0]->offset(0, 1);
      }
    }
    break;
  }
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
