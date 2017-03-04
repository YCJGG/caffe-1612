/*
# An inflation operation on feature maps with a learnable scaling factor. //revise
#
# Author: Wei Zhen @ IIE, CAS
#         Sun Yao @ IIE, CAS
# Create on: 2016-07-16
# Last modify: 2016-08-15
#
*/

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/interpolation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace caffe {

template <typename Dtype>
void InterpolationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // init inflation factor
    InflationFactorParameter inflation_param = this->layer_param().inflation_factor_param();
    
    // set margin
    margin_ = inflation_param.margin();
    CHECK_GT(margin_, 0);

    // if background mask is used, init factor_bg_mask_weight
    if (inflation_param.use_bg_mask() == true && bottom.size() == 4)
    {
	this->bg_mask_weight = inflation_param.bg_mask_weight();
    }
}

template <typename Dtype>
void InterpolationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // blob shape check
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                       << "corresponding to (num, channels, height, width)";

    // set factor
    const Dtype* bottom_factor = bottom[2]->cpu_data();
    
    // factor = bottom_factor[0], height = bottom_factor[1], f' = 8.1111/f.
    factor_value_ = (bottom[1]->height() - 1.0) / (bottom_factor[1] - 1.0) / bottom_factor[0];
    CHECK_LT(factor_value_, this->MAX_FACTOR);
    CHECK_GT(factor_value_, this->MIN_FACTOR);

    // calculate the top's shape 
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(), bottom[1]->width());
    factor_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[1]->height(), bottom[1]->width());
    
    // if background mask is used, init factor_bg_mask
    if (this->layer_param().inflation_factor_param().use_bg_mask() == true && bottom.size() == 4)
    {
	this->factor_bg_mask = float(bottom[3]->height()) / bottom[0]->height();
    }
}

template <typename Dtype>
void InterpolationLayer<Dtype>::inflate_forward(const Dtype *bottom_data, const int bottom_height, const int bottom_width, 
                                                Dtype *top_data, const int top_height, const int top_width, 
                                                const float factor, Dtype *factor_diff_matrix, const Dtype* label) {

    const float anchor_y = 0; //(bottom_height - 1) / 2.0;
    const float anchor_x = 0; //(bottom_width - 1) / 2.0;
    
    const float normalizer = margin_ * margin_ * margin_ * margin_;

    for (int y_t = 0; y_t < top_height; y_t++) {
        for (int x_t = 0; x_t < top_width; x_t++) {
            
            // coordinate on target map
            const int idx_t = y_t * top_width + x_t;
            
            top_data[idx_t] = 0;
            factor_diff_matrix[idx_t] = 0;
            
            float y_s = y_t / factor;
            float x_s = x_t / factor;
            CHECK_LT(y_s, bottom_height);
            CHECK_LT(x_s, bottom_width);
            
            for (int n = MAX(floor(y_s - margin_) + 1, 0); n < MIN(y_s + margin_, bottom_height); n++) {
                for (int m = MAX(floor(x_s - margin_) + 1, 0); m < MIN(x_s + margin_, bottom_width); m++) {
             
                    top_data[idx_t] += bottom_data[n * bottom_width + m] * (margin_ - abs(x_s - m)) * (margin_ - abs(y_s - n));
                    
                    factor_diff_matrix[idx_t] += bottom_data[n * bottom_width + m] 
                                                 * ((2 * (m >= x_s) - 1) * (margin_ - abs(y_s - n)) * (-(x_s - anchor_x) / factor)
                                                   +(2 * (n >= y_s) - 1) * (margin_ - abs(x_s - m)) * (-(y_s - anchor_y) / factor));
		    // when using background mask
		    if (label!= NULL && label[int(round(idx_t*this->factor_bg_mask))] == 0)
		    {
			factor_diff_matrix[idx_t] *= this->bg_mask_weight;
		    }
                }
            }
            // normalize
            top_data[idx_t] /= normalizer;
            factor_diff_matrix[idx_t] /= normalizer;
        }
    }
}

template <typename Dtype>
void InterpolationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* factor_diff_matrix = factor_diff_.mutable_cpu_data();
    const Dtype* label = NULL;
    if (this->layer_param().inflation_factor_param().use_bg_mask() == true)
	label = bottom[1]->cpu_data();
    
    // new shape
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();

    // resize
    for (int n = 0; n < num; n++) {  
        for (int c = 0; c < channels; c++) {
            const int index_in = (n * channels + c) * height * width;
            const int index_out = (n * channels + c) * top_height * top_width;
	    if (this->layer_param().inflation_factor_param().use_bg_mask() == true)
	    {
		const int index_label = n * bottom[1]->height() * bottom[1]->width();
            	inflate_forward(bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, factor_value_, factor_diff_matrix, label+index_label);
	    }
	    else
		inflate_forward(bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, factor_value_, factor_diff_matrix);
        }
    }
}

template <typename Dtype>
void InterpolationLayer<Dtype>::inflate_backward(Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
                                                 const Dtype *top_diff, const int top_height, const int top_width, 
                                                 const float factor) {
    const float normalizer = factor * factor * margin_ * margin_ * margin_ * margin_;
    
    for (int n = 0; n < bottom_height; n++) {
        for (int m = 0; m < bottom_width; m++) {
        
            // coordinate on target map
            const int idx_s = n * bottom_width + m;
            bottom_diff[idx_s] = 0;
            
            for (int y_t = MAX(floor((n - margin_) * factor) + 1, 0); y_t < MIN((n + margin_) * factor, top_height); y_t++) { 
                for (int x_t = MAX(floor((m - margin_) * factor) + 1, 0); x_t < MIN((m + margin_) * factor, top_width); x_t++) {
        
                    // diff
                    bottom_diff[idx_s] += top_diff[y_t * top_width + x_t] 
                                          * (margin_ - abs((x_t / factor) - m)) * (margin_ - abs((y_t / factor) - n));               
                }
            }
            // normalize
            bottom_diff[idx_s] /= normalizer;
        }
    }
}

template <typename Dtype>
void InterpolationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();
    
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* factor_diff = bottom[2]->mutable_cpu_diff();
    const Dtype* factor_diff_matrix = factor_diff_.cpu_data();

    if (propagate_down[0]) {
        // compute diff for bottom    
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                inflate_backward(bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, factor_value_);
            }
        }
    }

    if (propagate_down[1]) {
        LOG(FATAL) << this->type()
                   << " Layer cannot backpropagate to the second input.";
    }

    if (propagate_down[2]) {
        // compute diff for factor_
        // dL/d(factor) = sum(top.diff[i,j] * d(top.data[i,j])/d(factor))
        Dtype sum_dLoss_dfactor = caffe_cpu_dot(top[0]->count(), factor_diff_matrix,  top_diff);
        
        const Dtype* bottom_factor = bottom[2]->cpu_data();
    
        // factor = bottom_factor[0], height = bottom_factor[1], f' = 8.1111/f
        // dl/df = dl/df' * df'/df
        // dl/df' =  sum_dLoss_dfactor / num / factor_value_ / factor_value_);
        // df'/df = -8.11111/f/f
        *factor_diff = static_cast<Dtype>((sum_dLoss_dfactor / num / top_height / top_width) 
                        * (-(bottom[1]->height() - 1.0) / (bottom_factor[1] - 1.0) / bottom_factor[0] / bottom_factor[0]));
/*        LOG(INFO) << "  interpolation factor: " << factor_value_
                  << "  (" << height << " -> " << top_height << ")"
                  << "  f_diff: " << *factor_diff;
*/
    }          
}

#ifdef CPU_ONLY
STUB_GPU(InterpolationLayer);
#endif
INSTANTIATE_CLASS(InterpolationLayer);
REGISTER_LAYER_CLASS(Interpolation);
}  // namespace caffe
