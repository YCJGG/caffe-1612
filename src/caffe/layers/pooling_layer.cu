#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MaxPoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data, int* mask, Dtype* top_mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    Dtype maxval = -FLT_MAX;
    int maxidx = -1;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        if (bottom_slice[h * width + w] > maxval) {
          maxidx = h * width + w;
          maxval = bottom_slice[maxidx];
        }
      }
    }
    top_data[index] = maxval;
    if (mask) {
      mask[index] = maxidx;
    } else {
      top_mask[index] = maxidx;
    }
  }
}

template <typename Dtype>
__global__ void AvePoolForward(const int nthreads,
    const Dtype* const bottom_data, const int num, const int channels,
    const int height, const int width, const int pooled_height,
    const int pooled_width, const int kernel_h, const int kernel_w,
    const int stride_h, const int stride_w, const int pad_h, const int pad_w,
    Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    int hend = min(hstart + kernel_h, height + pad_h);
    int wend = min(wstart + kernel_w, width + pad_w);
    const int pool_size = (hend - hstart) * (wend - wstart);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    hend = min(hend, height);
    wend = min(wend, width);
    Dtype aveval = 0;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        aveval += bottom_slice[h * width + w];
      }
    }
    top_data[index] = aveval / pool_size;
  }
}

template <typename Dtype>
__global__ void StoPoolForwardTrain(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const rand_idx, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    Dtype cumsum = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
      }
    }
    const float thres = rand_idx[index] * cumsum;
    // Second pass: get value, and set index.
    cumsum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        if (cumsum >= thres) {
          rand_idx[index] = ((n * channels + c) * height + h) * width + w;
          top_data[index] = bottom_slice[h * width + w];
          return;
        }
      }
    }
  }
}


template <typename Dtype>
__global__ void StoPoolForwardTest(const int nthreads,
    const Dtype* const bottom_data,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    const int hstart = ph * stride_h;
    const int hend = min(hstart + kernel_h, height);
    const int wstart = pw * stride_w;
    const int wend = min(wstart + kernel_w, width);
    // We set cumsum to be 0 to avoid divide-by-zero problems
    Dtype cumsum = 0.;
    Dtype cumvalues = 0.;
    const Dtype* const bottom_slice =
        bottom_data + (n * channels + c) * height * width;
    // First pass: get sum
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        cumsum += bottom_slice[h * width + w];
        cumvalues += bottom_slice[h * width + w] * bottom_slice[h * width + w];
      }
    }
    top_data[index] = (cumsum > 0.) ? cumvalues / cumsum : 0.;
  }
}

// add on 2018-01-16, dense p_norm pooling forwarding
template <typename Dtype>
__global__ void DensePNormForward(const int nthreads,
	 const Dtype* padded_bottom_data, Dtype* top_data,
	 const Dtype* p_data, double* numerator_data, double* denominator_data,
	 const int bottom_num, const int channels,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels;
    const int n = index / pooled_width_ / pooled_height_ / channels;
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_bottom_height_);
    int wend = min(wstart + kernel_w_, padded_bottom_width_);
    double tmp_numerator = 0;
    double tmp_denominator = double(FLT_MIN);	// avoid divided by 0
    int top_idx = index;
    padded_bottom_data += (n * channels + c) * padded_bottom_height_ * padded_bottom_width_;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_bottom_width_ + w;
	double x_pow_p = (double)pow((double)padded_bottom_data[bottom_idx], (double)p_data[top_idx]);
	double x_pow_p_plus1 = x_pow_p * (double)padded_bottom_data[bottom_idx];
	tmp_numerator += x_pow_p_plus1;
	tmp_denominator += x_pow_p;
      }
    }
    top_data[top_idx] = (Dtype)(tmp_numerator / tmp_denominator);
    //avoid nan value
    //top_data[top_idx] = top_data[top_idx]!=top_data[top_idx] ? 0 : top_data[top_idx];
    numerator_data[top_idx] = tmp_numerator;
    denominator_data[top_idx] = tmp_denominator;
  }
}

// add on 2018-02-01, for dense p_norm pooling forwarding
template <typename Dtype>
__global__ void BottomPadding(const int nthreads,
	 const Dtype* bottom_data, Dtype* padded_bottom_data,
	 const int channels,
	 const int bottom_height_, const int bottom_width_,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pad_h_, const int pad_w_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int w = index % bottom_width_;
    const int h = (index / bottom_width_) % bottom_height_;
    const int c = (index / bottom_width_ / bottom_height_) % channels;
    const int n = index / bottom_width_ / bottom_height_ / channels;
    
    padded_bottom_data += (n * channels + c) * padded_bottom_height_ * padded_bottom_width_;
    padded_bottom_data[(h+pad_h_) * padded_bottom_width_ + (w+pad_w_)] = bottom_data[index];
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;
  Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->mutable_gpu_data();
    } else {
      mask = max_idx_.mutable_gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,
        mask, top_mask);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom_data, bottom[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    if (this->phase_ == TRAIN) {
      // We need to create the random index as well.
      caffe_gpu_rng_uniform(count, Dtype(0), Dtype(1),
                            rand_idx_.mutable_gpu_data());
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTrain<Dtype><<<CAFFE_GET_BLOCKS(count),
                                   CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_,
          rand_idx_.mutable_gpu_data(), top_data);
    } else {
      // NOLINT_NEXT_LINE(whitespace/operators)
      StoPoolForwardTest<Dtype><<<CAFFE_GET_BLOCKS(count),
                                  CAFFE_CUDA_NUM_THREADS>>>(
          count, bottom_data, bottom[0]->num(), channels_,
          height_, width_, pooled_height_, pooled_width_, kernel_h_,
          kernel_w_, stride_h_, stride_w_, top_data);
    }
    break;
  // add on 2018-01-16, dense p_norm pooling
  case PoolingParameter_PoolMethod_DENSE_P_NORM: {
    const Dtype* p_data = bottom[1]->gpu_data();
    // init numerator and denominator
    double* numerator_data = this->numerator.mutable_gpu_data();
    caffe_gpu_set(numerator.count(), double(0), numerator_data);
    double* denominator_data = this->denominator.mutable_gpu_data();
    caffe_gpu_set(denominator.count(), double(0), denominator_data);
    // bottom[0] padding
    Dtype* padded_bottom_data = padded_bottom.mutable_gpu_data();
    caffe_gpu_set(padded_bottom.count(), Dtype(0), padded_bottom_data);
    int bottom_count = bottom[0]->count();
    BottomPadding<<<CAFFE_GET_BLOCKS(bottom_count),
				 CAFFE_CUDA_NUM_THREADS>>>(
	bottom_count, bottom_data, padded_bottom_data,
	channels_, height_, width_, padded_height_, padded_width_,
	pad_h_, pad_w_);
    // The main loop
    DensePNormForward<Dtype><<<CAFFE_GET_BLOCKS(count),
				 CAFFE_CUDA_NUM_THREADS>>>(
	count, padded_bottom_data, top_data, p_data, numerator_data, denominator_data,
	bottom[0]->num(), channels_, padded_height_, padded_width_, pooled_height_, pooled_width_,
	kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);
    break;
  }
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


template <typename Dtype>
__global__ void MaxPoolBackward(const int nthreads, const Dtype* const top_diff,
    const int* const mask, const Dtype* const top_mask, const int num,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width, const int kernel_h,
    const int kernel_w, const int stride_h, const int stride_w, const int pad_h,
    const int pad_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart =
         (h + pad_h < kernel_h) ? 0 : (h + pad_h - kernel_h) / stride_h + 1;
    const int phend = min((h + pad_h) / stride_h + 1, pooled_height);
    const int pwstart =
         (w + pad_w < kernel_w) ? 0 : (w + pad_w - kernel_w) / stride_w + 1;
    const int pwend = min((w + pad_w) / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const int offset = (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice = top_diff + offset;
    if (mask) {
      const int* const mask_slice = mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    } else {
      const Dtype* const top_mask_slice = top_mask + offset;
      for (int ph = phstart; ph < phend; ++ph) {
        for (int pw = pwstart; pw < pwend; ++pw) {
          if (top_mask_slice[ph * pooled_width + pw] == h * width + w) {
            gradient += top_diff_slice[ph * pooled_width + pw];
          }
        }
      }
    }
    bottom_diff[index] = gradient;
  }
}

template <typename Dtype>
__global__ void AvePoolBackward(const int nthreads, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width + pad_w;
    const int h = (index / width) % height + pad_h;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        // figure out the pooling size
        int hstart = ph * stride_h - pad_h;
        int wstart = pw * stride_w - pad_w;
        int hend = min(hstart + kernel_h, height + pad_h);
        int wend = min(wstart + kernel_w, width + pad_w);
        int pool_size = (hend - hstart) * (wend - wstart);
        gradient += top_diff_slice[ph * pooled_width + pw] / pool_size;
      }
    }
    bottom_diff[index] = gradient;
  }
}


template <typename Dtype>
__global__ void StoPoolBackward(const int nthreads,
    const Dtype* const rand_idx, const Dtype* const top_diff,
    const int num, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % width;
    const int h = (index / width) % height;
    const int c = (index / width / height) % channels;
    const int n = index / width / height / channels;
    const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
    const int phend = min(h / stride_h + 1, pooled_height);
    const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
    const int pwend = min(w / stride_w + 1, pooled_width);
    Dtype gradient = 0;
    const Dtype* const rand_idx_slice =
        rand_idx + (n * channels + c) * pooled_height * pooled_width;
    const Dtype* const top_diff_slice =
        top_diff + (n * channels + c) * pooled_height * pooled_width;
    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        gradient += top_diff_slice[ph * pooled_width + pw] *
            (index == static_cast<int>(rand_idx_slice[ph * pooled_width + pw]));
      }
    }
    bottom_diff[index] = gradient;
  }
}

// add on 2018-01-16, dense p_norm pooling backwarding
template <typename Dtype>
__global__ void DensePNormBackward_P(const int nthreads,
	 const Dtype* padded_bottom_data, const Dtype* top_data, const Dtype* top_diff,
	 const Dtype* p_data, Dtype* p_diff,
	 const double* numerator_data, const double* denominator_data, const double* denominator_pow2_data,
	 const int bottom_num, const int channels,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels;
    const int n = index / pooled_width_ / pooled_height_ / channels;
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_bottom_height_);
    int wend = min(wstart + kernel_w_, padded_bottom_width_);
    // dL/dp_j = dL/dy_j * [sum_i(ln(x_i)*(x_i**(p_j+1))*denominator_j - sum_i(ln(x_i)*(x_i**p_j))*numerator_j] / (denominator_j ** 2)
    int top_idx = index;	// j of p_j, y_j
    double sum1 = 0.0;		// sum_i(ln(x_i)*(x_i**(p_j+1))
    double sum2 = 0.0;		// sum_i(ln(x_i)*(x_i**(p_j))
    int bottom_offset = (n * channels + c) * padded_bottom_height_ * padded_bottom_width_;
    padded_bottom_data += bottom_offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_bottom_width_ + w;
	double x_pow_p = (double)pow((double)padded_bottom_data[bottom_idx], (double)p_data[top_idx]);
	double x_pow_p_plus1 = x_pow_p * (double)padded_bottom_data[bottom_idx];
	// avoid x->0 in log(.)
	double bottom_data_value = (double)padded_bottom_data[bottom_idx]<1e-3 ? (double)1e-3 : (double)padded_bottom_data[bottom_idx];
	sum1 += (double)log(bottom_data_value) * x_pow_p_plus1;
	sum2 += (double)log(bottom_data_value) * x_pow_p;
//printf("%f,%f,%f,%f,%f,%f,%f\n",(double)log(bottom_data_value) * x_pow_p_plus1,(double)log(bottom_data_value) * x_pow_p,(double)padded_bottom_data[bottom_idx],bottom_data_value,numerator_data[top_idx],denominator_data[top_idx],top_diff[top_idx]);
      }
    }
    double tmp = sum1/(denominator_data[top_idx]+(double)1e-34) - sum2*(numerator_data[top_idx] / (denominator_data[top_idx]+(double)1e-34) / (denominator_data[top_idx]+(double)1e-34));
    //double tmp = (sum1*denominator_data[top_idx] - sum2*numerator_data[top_idx])  / (denominator_data[top_idx]+(double)1e-34) / (denominator_data[top_idx]+(double)1e-34);
    p_diff[top_idx] = top_diff[top_idx] * (Dtype)tmp;
  }
}

// add on 2018-01-31, dense p_norm pooling backwarding
template <typename Dtype>
__global__ void DensePNormBackward_data(const int nthreads,
	 const Dtype* padded_bottom_data, Dtype* bottom_diff, const Dtype* top_data, const Dtype* top_diff,
	 const Dtype* p_data,
	 const double* numerator_data, const double* denominator_data, const double* denominator_pow2_data,
	 const int bottom_num, const int channels,
	 const int bottom_height_, const int bottom_width_,
	 const int padded_bottom_height_, const int padded_bottom_width_,
	 const int pooled_height_, const int pooled_width_,
	 const int kernel_h_, const int kernel_w_,
	 const int stride_h_, const int stride_w_,
	 const int pad_h_, const int pad_w_) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    // find out the local index
    // find out the local offset
    const int w = index % padded_bottom_width_;
    const int h = (index / padded_bottom_width_) % padded_bottom_height_;
    const int c = (index / padded_bottom_width_ / padded_bottom_height_) % channels;
    const int n = index / padded_bottom_width_ / padded_bottom_height_ / channels;
    // find out related top idx
    const int phstart = (h < kernel_h_) ? 0 : (h - kernel_h_) / stride_h_ + 1;
    const int phend = min(h / stride_h_ + 1, pooled_height_);
    const int pwstart = (w < kernel_w_) ? 0 : (w - kernel_w_) / stride_w_ + 1;
    const int pwend = min(w / stride_w_ + 1, pooled_width_);
    // dL/dx_i = dL/dy_j * [((p_j+1)*(x_i**p_j)*denominator_j) - (p_j*(x_i**(p_j-1))*numerator_j)] / (denominator_j ** 2)
    int bottom_idx = index;	// i of x_i, namely (n, c, h, w)
    bottom_diff += (n * channels + c) * bottom_height_ * bottom_width_;
    int top_offset = (n * channels + c) * pooled_height_ * pooled_width_;
    top_diff += top_offset;
    p_data += top_offset;
    numerator_data += top_offset;
    denominator_data += top_offset;
    denominator_pow2_data += top_offset;
    for (int ph = phstart; ph < phend; ++ph) {		// sum up gradients w.r.t j
      for (int pw = pwstart; pw < pwend; ++pw) {
	// dL/dx_i
	// directly crop out from padded gradient via idx transform
	if ((h >= pad_h_) && (w >= pad_w_) && (h < bottom_height_ + pad_h_) && (w < bottom_width_ + pad_w_))
	{
		int top_idx = ph * pooled_width_ + pw;		// j of p_j, y_j
		double x_pow_p_minus1 = (double)pow((double)padded_bottom_data[bottom_idx]+1e-10, (double)p_data[top_idx]-1);
		double x_pow_p = (double)pow((double)padded_bottom_data[bottom_idx], (double)p_data[top_idx]);

		int ori_bottom_idx = (h-pad_h_) * bottom_width_ + (w-pad_w_);
		double tmp = ( ((double)(p_data[top_idx]+1) * x_pow_p * denominator_data[top_idx])
		   - ((double)p_data[top_idx] * x_pow_p_minus1 * numerator_data[top_idx]) )
	 	  / (denominator_data[top_idx]+(double)1e-34) / (denominator_data[top_idx]+(double)1e-34);
		bottom_diff[ori_bottom_idx] += top_diff[top_idx] * (Dtype)tmp;
	}
      }
    }
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  caffe_gpu_set(count, Dtype(0.), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    if (use_top_mask) {
      top_mask = top[1]->gpu_data();
    } else {
      mask = max_idx_.gpu_data();
    }
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, mask, top_mask, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_,
        kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
        bottom_diff);
    break;
  case PoolingParameter_PoolMethod_AVE:
    // NOLINT_NEXT_LINE(whitespace/operators)
    AvePoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, top[0]->num(), channels_,
        height_, width_, pooled_height_, pooled_width_, kernel_h_,
        kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, bottom_diff);
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    // NOLINT_NEXT_LINE(whitespace/operators)
    StoPoolBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, rand_idx_.gpu_data(), top_diff,
        top[0]->num(), channels_, height_, width_, pooled_height_,
        pooled_width_, kernel_h_, kernel_w_, stride_h_, stride_w_,
        bottom_diff);
    break;
  // add on 2018-01-16, dense p_norm pooling
  case PoolingParameter_PoolMethod_DENSE_P_NORM:  {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const Dtype* padded_bottom_data = padded_bottom.gpu_data();
    const Dtype* top_data = top[0]->gpu_data();
    Dtype* p_diff = bottom[1]->mutable_gpu_diff();
    const Dtype* p_data = bottom[1]->gpu_data();
    // p_diff are init with 0 in solver.cpp::Step()
    const int top_count = top[0]->count();
    const int padded_bottom_count = padded_bottom.count();
    // init numerator and denominator
    const double* numerator_data = this->numerator.gpu_data();
    const double* denominator_data = this->denominator.gpu_data();
    // get denominator**2 in advance
    Blob<double> denominator_pow2;
    denominator_pow2.ReshapeLike(denominator);
    double* denominator_pow2_data = denominator_pow2.mutable_gpu_data();
    //caffe_gpu_powx(denominator.count(), denominator_data, double(2), denominator_pow2_data);
    // The main loop
    // gradients w.r.t. p, loop with top's idx
    DensePNormBackward_P<Dtype><<<CAFFE_GET_BLOCKS(top_count), CAFFE_CUDA_NUM_THREADS>>>(
	top_count, padded_bottom_data, top_data, top_diff, p_data, p_diff,
	numerator_data, denominator_data, denominator_pow2_data,
	bottom[0]->num(), channels_, padded_height_, padded_width_, pooled_height_, pooled_width_,
	kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);


/*    Dtype* p_diff_cpu = bottom[1]->mutable_cpu_diff();
    Dtype* p_data_cpu = bottom[1]->mutable_cpu_data();
    double* denominator_pow2_data_cpu = denominator_pow2.mutable_cpu_data();
    const Dtype* top_diff_cpu = top[0]->cpu_diff();
    const Dtype* padded_bottom_data_cpu = padded_bottom.cpu_data();
    const double* numerator_data_cpu = this->numerator.cpu_data();
    const double* denominator_data_cpu = this->denominator.cpu_data();
    double* sum1_cpu = sum1.mutable_cpu_data();
    double* sum2_cpu = sum1.mutable_cpu_data();
	for (int i = 0; i < bottom[1]->count(); i++){
		if (isnan(p_diff_cpu[i])){//top[0]->cpu_data()[i]>1e4 {
    const int pw = i % pooled_width_;
    const int ph = (i / pooled_width_) % pooled_height_;
    const int c = (i / pooled_width_ / pooled_height_) % channels_;
    const int n = i / pooled_width_ / pooled_height_ / channels_;
if (n>=2 || pw%4!=0 || ph%4!=0)	continue;
// check log file size
FILE *fp=fopen("/home/kfxw/ProjectFiles/Python_scripts/Shift_variant_pooling/scripts/log4.log","r");  
fseek(fp,0L,SEEK_END);  
if (ftell(fp)>536870912) break; 
fclose(fp);
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_height_);
    int wend = min(wstart + kernel_w_, padded_width_);
			LOG(INFO)<< '('<<n<<','<<c<<','<<ph<<','<<pw<<")";
    for (int h = hstart; h < hend; ++h) {		// sum up gradients w.r.t j
      for (int w = wstart; w < wend; ++w) {
			LOG(INFO)<< padded_bottom_data_cpu[(n*channels_+c)*padded_height_*padded_width_+h*padded_width_+w];
	}
    }
			LOG(INFO)<< bottom[1]->count()<<','<<denominator_pow2_data_cpu[i]<<','<<p_data_cpu[i]<<','<<top_diff_cpu[i]<<','<<numerator_data_cpu[i]<<','<<denominator_data_cpu[i]<<','<<sum1_cpu[1]<<','<<sum2_cpu[i]<<','<<top[0]->cpu_data()[i];
		}
	}
*/



	Dtype* p_diff_cpu = bottom[1]->mutable_cpu_diff();
	for (int index = 0; index < bottom[1]->count(); index++){
if (isnan(p_diff_cpu[index]) || isinf(p_diff_cpu[index])){
    const Dtype* padded_bottom_data_cpu = padded_bottom.cpu_data();
    const double* numerator_data_cpu = this->numerator.cpu_data();
    const double* denominator_data_cpu = this->denominator.cpu_data();
    const int pw = index % pooled_width_;
    const int ph = (index / pooled_width_) % pooled_height_;
    const int c = (index / pooled_width_ / pooled_height_) % channels_;
    const int n = index / pooled_width_ / pooled_height_ / channels_;
    int hstart = ph * stride_h_;
    int wstart = pw * stride_w_;
    int hend = min(hstart + kernel_h_, padded_height_);
    int wend = min(wstart + kernel_w_, padded_width_);
    int top_idx = index;	// j of p_j, y_j
    double sum1 = 0.0;		// sum_i(ln(x_i)*(x_i**(p_j+1))
    double sum2 = 0.0;		// sum_i(ln(x_i)*(x_i**(p_j))
    int bottom_offset = (n * channels_ + c) * padded_height_ * padded_width_;
    padded_bottom_data_cpu += bottom_offset;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_width_ + w;
	double x_pow_p = (double)pow((double)padded_bottom_data_cpu[bottom_idx], (double)bottom[1]->mutable_cpu_data()[top_idx]);
	double x_pow_p_plus1 = x_pow_p * (double)padded_bottom_data_cpu[bottom_idx];
	double bottom_data_value = (double)padded_bottom_data_cpu[bottom_idx]<1e-3 ? (double)1e-3 : (double)padded_bottom_data_cpu[bottom_idx];
	sum1 += (double)log(bottom_data_value) * x_pow_p_plus1;
	sum2 += (double)log(bottom_data_value) * x_pow_p;
      }
    }
    double tmp = sum1/(denominator_data_cpu[top_idx]+(double)1e-34) - sum2*(numerator_data_cpu[top_idx] / (denominator_data_cpu[top_idx]+(double)1e-34) / (denominator_data_cpu[top_idx]+(double)1e-34));
    //p_diff[top_idx] = top_diff[top_idx] * (Dtype)tmp;
    LOG(INFO)<<"--------------------";
    LOG(INFO)<<"p_diff: "<<p_diff_cpu[top_idx]<<"\ttop_diff: "<<top[0]->cpu_diff()[top_idx];
    LOG(INFO)<<"(Dtype)tmp: "<<(Dtype)tmp<<"\ttmp: "<<tmp;
    LOG(INFO)<<"tmp term1: "<<sum1/(denominator_data_cpu[top_idx]+(double)1e-34)<<"\ttmp term2: "<<sum2*(numerator_data_cpu[top_idx] / (denominator_data_cpu[top_idx]+(double)1e-34) / (denominator_data_cpu[top_idx]+(double)1e-34));
    LOG(INFO)<<"sum1: "<<sum1<<"\tsum2: "<<sum2<<"\tnumerator: "<<numerator_data_cpu[top_idx]<<"\tdenominator: "<<denominator_data_cpu[top_idx];
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
	int bottom_idx = h * padded_width_ + w;
	double x_pow_p = (double)pow((double)padded_bottom_data_cpu[bottom_idx], (double)bottom[1]->mutable_cpu_data()[top_idx]);
	double x_pow_p_plus1 = x_pow_p * (double)padded_bottom_data_cpu[bottom_idx];
	double bottom_data_value = (double)padded_bottom_data_cpu[bottom_idx]<1e-3 ? (double)1e-3 : (double)padded_bottom_data_cpu[bottom_idx];
	LOG(INFO)<<"ori_data: "<<padded_bottom_data_cpu[bottom_idx]<<"\tbottom_data: "<<bottom_data_value;
	LOG(INFO)<<"x_pow_p: "<<x_pow_p<<"\tx_pow_p_plus1: "<<x_pow_p_plus1<<"\tp: "<<bottom[1]->mutable_cpu_data()[top_idx];
	LOG(INFO)<<"delta sum1: "<<(double)log(bottom_data_value) * x_pow_p_plus1<<"\tdelta sum2: "<<(double)log(bottom_data_value) * x_pow_p;
      }
    }
}
		/*if (isnan(p_diff_cpu[i])){
			p_diff_cpu[i] = 0;
			LOG(INFO)<<"p diff is nan at idx "<<i<<"/count "<<bottom[1]->count();
		} else if (isinf(p_diff_cpu[i])){
			p_diff_cpu[i] = 0;
			LOG(INFO)<<"p diff is inf at idx "<<i<<"/count "<<bottom[1]->count();
		}*/
	}




    // gradients w.r.t. bottom[0], loop with bottom's idx
    DensePNormBackward_data<Dtype><<<CAFFE_GET_BLOCKS(padded_bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
	padded_bottom_count, padded_bottom_data, bottom_diff, top_data, top_diff, p_data,
	numerator_data, denominator_data, denominator_pow2_data,
	bottom[0]->num(), channels_, height_, width_, padded_height_, padded_width_, pooled_height_, pooled_width_,
	kernel_h_, kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_);
    // p_diff gradient scaling
    caffe_gpu_scal(bottom[1]->count(), (Dtype)100, p_diff);
    break;
  }
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PoolingLayer);


}  // namespace caffe
