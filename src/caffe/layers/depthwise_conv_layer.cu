#include <vector>
#include <algorithm>
#include <cfloat>
#include "caffe/layers/depthwise_conv_layer.hpp"
#include "caffe/util/math_functions.hpp"


namespace caffe {

__device__ float atomicAddme(float* address, float val)
{
    return atomicAdd(address,val);
}

__device__ double atomicAddme(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseWeightForward(const int nthreads,
    const Dtype* const bottom_data, const Dtype* const weight_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / top_height / top_width;
    const int c = (index / top_height / top_width) % channels;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_in = -pad_h + h * stride_h + kh * dilation_h;
        const int w_in = -pad_w + w * stride_w + kw * dilation_w;
        if ((h_in >= 0) && (h_in < bottom_height)
              && (w_in >= 0) && (w_in < bottom_width)) {
          const int offset = ((n * channels + c) * bottom_height + h_in)
                * bottom_width + w_in;
          value += (*weight) * bottom_data[offset];
        }
        ++weight;
      }
    }
    top_data[index] = value;
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBiasForward(const int nthreads,
    const Dtype* const bias_data, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = (index / top_height / top_width) % channels;
    top_data[index] += bias_data[c];
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
      const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight_data = this->blobs_[0]->gpu_data();
  const int count = top[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  ConvolutionDepthwiseWeightForward<Dtype>
        // NOLINT_NEXT_LINE(whitespace/operators)
        <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, weight_data, num, channels,
      top_height, top_width, bottom_height, bottom_width,
      kernel_h_, kernel_w_, stride_h_, stride_w_,
      pad_h_, pad_w_, dilation_h_, dilation_w_, top_data);
  if (this->layer_param_.convolution_param().bias_term()) {
    const Dtype* bias_data = this->blobs_[1]->gpu_data();
    ConvolutionDepthwiseBiasForward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bias_data, num, channels,
        top_height, top_width, top_data);
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseWeightBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const bottom_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, Dtype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int kh = (index / kernel_w / num / top_height / top_width)
          % kernel_h;
    const int kw = (index / num / top_height / top_width) % kernel_w;
    const int h_in = -pad_h + h * stride_h + kh * dilation_h;
    const int w_in = -pad_w + w * stride_w + kw * dilation_w;
    if ((h_in >= 0) && (h_in < bottom_height)
          && (w_in >= 0) && (w_in < bottom_width)) {
      const int c = index / kernel_h / kernel_w / num / top_height / top_width;
      const int n = (index / top_height / top_width) % num;
      const int top_offset = ((n * channels + c) * top_height + h)
            * top_width + w;
      const int bottom_offset = ((n * channels + c) * bottom_height + h_in)
            * bottom_width + w_in;
      buffer_data[index] = top_diff[top_offset] * bottom_data[bottom_offset];
    } else {
      buffer_data[index] = 0;
    }
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBottomBackward(const int nthreads,
    const Dtype* const top_diff, const Dtype* const weight_data,
    const int num, const int channels, const int top_height,
    const int top_width, const int bottom_height, const int bottom_width,
    const int kernel_h, const int kernel_w, const int stride_h,
    const int stride_w, const int pad_h, const int pad_w,
    const int dilation_h, const int dilation_w, Dtype* const bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / channels / bottom_height / bottom_width;
    const int c = (index / bottom_height / bottom_width) % channels;
    const int h = (index / bottom_width) % bottom_height;
    const int w = index % bottom_width;
    const Dtype* weight = weight_data + c * kernel_h * kernel_w;
    Dtype value = 0;
    for (int kh = 0; kh < kernel_h; ++kh) {
      for (int kw = 0; kw < kernel_w; ++kw) {
        const int h_out_s = h + pad_h - kh * dilation_h;
        const int w_out_s = w + pad_w - kw * dilation_w;
        if (((h_out_s % stride_h) == 0) && ((w_out_s % stride_w) == 0)) {
          const int h_out = h_out_s / stride_h;
          const int w_out = w_out_s / stride_w;
          if ((h_out >= 0) && (h_out < top_height)
                && (w_out >= 0) && (w_out < top_width)) {
            const int offset = ((n * channels + c) * top_height + h_out)
                  * top_width + w_out;
            value += (*weight) * top_diff[offset];
          }
        }
        ++weight;
      }
    }
    bottom_diff[index] += value;
  }
}

template <typename Dtype>
__global__ void ConvolutionDepthwiseBiasBackward(const int nthreads,
    const Dtype* const top_diff, const int num, const int channels,
    const int top_height, const int top_width, Dtype* const buffer_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int c = index / num / top_height / top_width;
    const int n = (index / top_height / top_width) % num;
    const int h = (index / top_width) % top_height;
    const int w = index % top_width;
    const int offset = ((n * channels + c) * top_height + h) * top_width + w;
    buffer_data[index] = top_diff[offset];
  }
}

template <typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
      const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
      const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const int bottom_count = bottom[0]->count();
  const int num = top[0]->num();
  const int channels = top[0]->channels();
  const int top_height = top[0]->height();
  const int top_width = top[0]->width();
  const int bottom_height = bottom[0]->height();
  const int bottom_width = bottom[0]->width();
  const int length = num * top_height * top_width;
  caffe_gpu_set(bottom_count, Dtype(0), bottom[0]->mutable_gpu_diff());
  if (this->layer_param_.convolution_param().bias_term()
        && this->param_propagate_down_[1]) {
    const int bias_buffer_count = bias_buffer_.count();
    Dtype* bias_buffer_mutable_data = bias_buffer_.mutable_gpu_data();
    ConvolutionDepthwiseBiasBackward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(bias_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        bias_buffer_count, top_diff, num, channels,
        top_height, top_width, bias_buffer_mutable_data);
    const int bias_count = this->blobs_[1]->count();
    const Dtype* bias_buffer_data = bias_buffer_.gpu_data();
    Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
    const Dtype* bias_multiplier_data = bias_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasNoTrans, bias_count, length, Dtype(1),
          bias_buffer_data, bias_multiplier_data, Dtype(1), bias_diff);
  }
  if (this->param_propagate_down_[0]) {
    const int weight_buffer_count = weight_buffer_.count();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* weight_buffer_mutable_data = weight_buffer_.mutable_gpu_data();
    ConvolutionDepthwiseWeightBackward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(weight_buffer_count), CAFFE_CUDA_NUM_THREADS>>>(
        weight_buffer_count, top_diff, bottom_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, weight_buffer_mutable_data);
    const int weight_count = this->blobs_[0]->count();
    const Dtype* weight_buffer_data = weight_buffer_.gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    const Dtype* weight_multiplier_data = weight_multiplier_.gpu_data();
    caffe_gpu_gemv(CblasNoTrans, weight_count, length, Dtype(1),
          weight_buffer_data, weight_multiplier_data, Dtype(1), weight_diff);
  }
  if (propagate_down[0]) {
    const Dtype* weight_data = this->blobs_[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    ConvolutionDepthwiseBottomBackward<Dtype>
          // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, top_diff, weight_data, num, channels,
        top_height, top_width, bottom_height, bottom_width,
        kernel_h_, kernel_w_, stride_h_, stride_w_,
        pad_h_, pad_w_, dilation_h_, dilation_w_, bottom_diff);
  }
}

/*template <typename Dtype>
__global__ void ConvForward(const int nthreads,
		const Dtype* const bottom_data, const int num, const int channels,
		const int height, const int width,const int conved_height,
		const int conved_width,const int kernel_h, const int kernel_w,
		const int stride_h, const int stride_w, const int pad_h, const int pad_w,
		Dtype* const top_data,const Dtype* const weight,const Dtype* const bias,const bool bias_term_) {
	CUDA_KERNEL_LOOP(index, nthreads) {

		const int pw = index % conved_width;
		const int ph = (index / conved_width) % conved_height;
		const int c = (index / conved_width / conved_height) % channels;
		const int n = index / conved_width / conved_height / channels;
		int hstart = ph * stride_h - pad_h;
		int wstart = pw * stride_w - pad_w;
		int hend = min(hstart + kernel_h, height + pad_h);
		int wend = min(wstart + kernel_w, width + pad_w);
		hstart = max(hstart, 0);
		wstart = max(wstart, 0);
		hend = min(hend, height);
		wend = min(wend, width);
		Dtype aveval = 0;
		const Dtype* const bottom_slice =
		bottom_data + (n * channels + c) * height * width;
		const Dtype* const weight_slice =
		weight + c * kernel_h * kernel_w;

		int khstart=hend<kernel_h?kernel_h-hend:0;
		int kwstart=wend<kernel_w?kernel_w-wend:0;
		for (int h = hstart; h < hend; ++h) {
			for (int w = wstart; w < wend; ++w) {
				aveval += bottom_slice[h * width + w]*weight_slice[(khstart+h-hstart) * kernel_w + (kwstart+w-wstart)];
			}
		}
		if(bias_term_) {
			aveval+=bias[c];
		}
		top_data[index] = aveval;
	}
}

template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Forward_gpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	const Dtype* weight = this->blobs_[0]->gpu_data();
	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	for (int i = 0; i < bottom.size(); ++i) {
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* top_data = top[i]->mutable_gpu_data();
		const int count = top[i]->count();
		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

		const int kernel_h_ = kernel_shape_data[0];
		const int kernel_w_ = kernel_shape_data[1];
		const int stride_h_ = stride_data[0];
		const int stride_w_ = stride_data[1];
		const int pad_h_ = pad_data[0];
		const int pad_w_ = pad_data[1];

		const int conved_height = this->output_shape_[0];
		const int conved_weight = this->output_shape_[1];

		const bool bias_term_ = this->bias_term_;

		if (bias_term_) {
			const Dtype* const bias = this->blobs_[1]->gpu_data();
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,bias,bias_term_);
		} else {
			ConvForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
					count, bottom_data, bottom[i]->num(), channels_,
					height_, width_,conved_height,conved_weight,kernel_h_,
					kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, top_data,weight,0,bias_term_);
		}
	}
}

template <typename Dtype>
__global__ void ConvBackward(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const bottom_diff,
const Dtype* const weight) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int w = index % width + pad_w;
		const int h = (index / width) % height + pad_h;
		const int c = (index / width / height) % channels;
		const int n = index / width / height / channels;
		
		const int phstart = (h < kernel_h) ? 0 : (h - kernel_h) / stride_h + 1;
		const int phend = min(h / stride_h + 1, conved_height);
		const int pwstart = (w < kernel_w) ? 0 : (w - kernel_w) / stride_w + 1;
		const int pwend = min(w / stride_w + 1, conved_width);
		
		const int khstart=(h >= kernel_h) ? ((h-kernel_h)%stride_h)+(kernel_h-stride_h): h;
		const int kwstart=(w >= kernel_w) ? ((w-kernel_w)%stride_w)+(kernel_w-stride_w) : w;
		
		Dtype gradient = 0;
		const Dtype* const top_diff_slice =
		top_diff + (n * channels + c) * conved_height * conved_width;
		
		const Dtype* const weight_slice =weight + c * kernel_h * kernel_w;
		for (int ph = phstart; ph < phend; ++ph) {
			for (int pw = pwstart; pw < pwend; ++pw) {
				int kh=khstart-(ph-phstart)*stride_h;
				int kw=kwstart-(pw-pwstart)*stride_w;
				gradient += top_diff_slice[ph * conved_width + pw] *weight_slice[kh*kernel_w+kw];
			}
		}
		bottom_diff[index] = gradient;
	}
}

#define DIVIDE_CEIL(a,b) a/b+((a/b*b)<a)


template <typename Dtype>
__global__ void ConvBackwardWeight(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const weight_diff,
const Dtype* const bottom_data) {

	CUDA_KERNEL_LOOP(index, nthreads) {
		const int kw=index % kernel_w;
		const int kh= (index /kernel_w)%kernel_h;
		const int c=index /kernel_w/kernel_h;
		Dtype gradient = 0;
		for( int n=0;n<num;n++) {
			
			const Dtype* const top_diff_slice = top_diff + (n * channels + c) * conved_height * conved_width;
			const Dtype* const bottom_data_slice = bottom_data + (n * channels + c) * height * width;
		
			
			const int phstart=max(DIVIDE_CEIL((pad_h-kh),stride_h),0);
			const int phend=min(DIVIDE_CEIL((height+pad_h-kh),stride_h),conved_height);
		
			const int pwstart=max(DIVIDE_CEIL((pad_w-kw),stride_w),0);
			
			const int pwend=min(DIVIDE_CEIL((width+pad_w-kw),stride_w),conved_width);
			for(int ph=phstart;ph<phend;ph++){
				for (int pw=pwstart;pw<pwend;pw++){
					const int h=ph*stride_h+kh-pad_h;
					const int w=pw*stride_w+kw-pad_w;
					gradient+=top_diff_slice[ph * conved_width + pw]*bottom_data_slice[h*width+w];
				}
			}
		}
		weight_diff[c * kernel_h * kernel_w+kh*kernel_w+kw]+=gradient;
	}
}

template <typename Dtype>
__global__ void ConvBackwardBias(const int nthreads,
const Dtype* const top_diff,
const int num, const int channels, const int height,
const int width, const int conved_height, const int conved_width,
const int kernel_h, const int kernel_w, const int stride_h,
const int stride_w, const int pad_h, const int pad_w,
Dtype* const bias_diff) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int c = index;
		Dtype gradient=0;
		for( int n=0;n<num;n++) {
			const Dtype* const top_diff_slice =
			top_diff + (n * channels + c) * conved_height * conved_width;
			for(int ph=0;ph<conved_height;ph++) {
				for (int pw=0;pw<conved_width;pw++) {
					gradient+=top_diff_slice[ph * conved_width + pw];
				}
			}
		}
		bias_diff[c]+=gradient;
	}
}
template<typename Dtype>
void DepthwiseConvolutionLayer<Dtype>::Backward_gpu(
const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
const vector<Blob<Dtype>*>& bottom) {


	int* kernel_shape_data = this->kernel_shape_.mutable_cpu_data();
	int* stride_data = this->stride_.mutable_cpu_data();
	int* pad_data = this->pad_.mutable_cpu_data();

	const Dtype* weight = this->blobs_[0]->gpu_data();
	Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();

	const bool bias_term_ = this->bias_term_;
	Dtype* bias_diff = bias_term_ ? this->blobs_[1]->mutable_gpu_diff() : 0;
	const bool bias_propagate_down_ = this->param_propagate_down_[1];
	const bool weight_propagate_down_ = this->param_propagate_down_[0];


	const int kernel_h_ = kernel_shape_data[0];
	const int kernel_w_ = kernel_shape_data[1];
	const int stride_h_ = stride_data[0];
	const int stride_w_ = stride_data[1];
	const int pad_h_ = pad_data[0];
	const int pad_w_ = pad_data[1];

	const int conved_height = this->output_shape_[0];
	const int conved_weight = this->output_shape_[1];

	for (int i = 0; i < top.size(); ++i) {

		const Dtype* top_diff = top[i]->gpu_diff();
		const Dtype* bottom_data = bottom[i]->gpu_data();
		Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();

		vector<int> shape_ = bottom[i]->shape();
		const int channels_ = shape_[1];
		const int height_ = shape_[2];
		const int width_ = shape_[3];

		// Bias gradient, if necessary.
		if (bias_term_ && bias_propagate_down_) {
			const int count_bias = channels_;
			ConvBackwardBias<Dtype><<<CAFFE_GET_BLOCKS(count_bias), CAFFE_CUDA_NUM_THREADS>>>(
				count_bias, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				bias_diff);
		}
		// gradient w.r.t. weight. Note that we will accumulate diffs.
		if (weight_propagate_down_) {
			const int count_weight = channels_ * kernel_h_ * kernel_w_;
			ConvBackwardWeight<Dtype><<<CAFFE_GET_BLOCKS(count_weight), CAFFE_CUDA_NUM_THREADS>>>(
					count_weight, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_,
				weight_diff,
				bottom_data);
		}
		// gradient w.r.t. bottom data, if necessary.
		if (propagate_down[i]) {
			const int count_bottom=bottom[i]->count();
			ConvBackward<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
				count_bottom, top_diff, bottom[i]->num(), channels_,
				height_, width_,conved_height,conved_weight,kernel_h_,
				kernel_w_, stride_h_, stride_w_, pad_h_, pad_w_, 
				bottom_diff,
				weight);
		}
	}

}*/


INSTANTIATE_LAYER_GPU_FUNCS (DepthwiseConvolutionLayer);

}  // namespace caffe

