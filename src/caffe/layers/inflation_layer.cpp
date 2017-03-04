/*
# An inflation operation on feature maps with a learnable scaling factor.
#
# Author: Wei Zhen @ IIE, CAS
#         Sun Yao @ IIE, CAS
# Create on: 2016-07-16
# Last modify: 2016-11-01
#
*/

#include <algorithm>
#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/layers/inflation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace caffe {

template <typename Dtype>
void InflationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

    // init inflation factor
    InflationFactorParameter inflation_param = this->layer_param().inflation_factor_param();
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob<Dtype>(vector<int>(0)));

    shared_ptr<Filler<Dtype> > filler;
    FillerParameter filler_param;
    filler_param.set_type("constant");
    filler_param.set_value(inflation_param.factor());
    filler.reset(GetFiller<Dtype>(filler_param));
    filler->Fill(this->blobs_[0].get());

    const Dtype* factor_ = this->blobs_[0]->cpu_data();
    // check factor's value range
    CHECK_LE(*factor_, this->MAX_FACTOR);
    CHECK_GE(*factor_, this->MIN_FACTOR);
    
    // set from where factor is learned.
    iter_counter_ = 0;
    
    // set margin
    margin_ = inflation_param.margin();
    CHECK_GT(margin_, 0);

    // if background mask is used, init factor_bg_mask_weight
    if (inflation_param.use_bg_mask() == true && bottom.size() == 2)
    {
	this->bg_mask_weight = inflation_param.bg_mask_weight();
    }
}

template <typename Dtype>
void InflationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    // blob shape check
    CHECK_EQ(4, bottom[0]->num_axes()) << "Input must have 4 axes, "
                                       << "corresponding to (num, channels, height, width)";

    // calculate the top's shape
    // top shape = (bottom shape - 1) * factor + 1
    Dtype* factor_ = this->blobs_[0]->mutable_cpu_data();
    CHECK_GT(*factor_, 0);
        
    // check and correct inflation factor before forwarding
    if (*factor_ > this->MAX_FACTOR) {
        *factor_ = this->MAX_FACTOR;
        LOG(INFO) << "Warning: Inflation factor " << *factor_
                  << "exceeds its upper bound " << this->MAX_FACTOR
                  << "and has been corrected.";
    }
    if (*factor_ < this->MIN_FACTOR) {
        *factor_ = this->MIN_FACTOR;
        LOG(INFO) << "Warning: Inflation factor " << *factor_
                  << "exceeds its lower bound " << this->MIN_FACTOR
                  << "and has been corrected.";
    }        
        
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = round((height - 1) * *factor_ + 1);
    const int top_width = round((width - 1) * *factor_ + 1);
    
    CHECK_LE(*factor_, this->MAX_FACTOR);
    CHECK_GE(*factor_, this->MIN_FACTOR);
    CHECK_GT(top_height, 0);
    CHECK_GT(top_width, 0);
        
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), top_height, top_width);
    factor_diff_.Reshape(bottom[0]->num(), bottom[0]->channels(), top_height, top_width);        
        
    // resize top[1] = (factor, height)
    top[1]->Reshape(1, 1, 1, 2);
    
    Dtype* top_factor = top[1]->mutable_cpu_data();    
    top_factor[0] = *factor_;
    top_factor[1] = height;

    // if background mask is used, init factor_bg_mask
    if (this->layer_param().inflation_factor_param().use_bg_mask() == true && bottom.size() == 2)
    {
	this->factor_bg_mask = float(bottom[1]->height()) / bottom[0]->height();
    }
}

template <typename Dtype>
void InflationLayer<Dtype>::inflate_forward(const Dtype *bottom_data, const int bottom_height, const int bottom_width, 
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
void InflationLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    
    const Dtype* factor_ = this->blobs_[0]->cpu_data();

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
	label = bottom[3]->cpu_data();

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
		const int index_label = n * bottom[3]->height() * bottom[3]->width();
                inflate_forward(bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix, label+index_label);
	    }
	    else
		inflate_forward(bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix, NULL);
        }
    }
}

template <typename Dtype>
void InflationLayer<Dtype>::inflate_backward(Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
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
void InflationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
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
    const Dtype* factor_ = this->blobs_[0]->cpu_data();
    Dtype* factor_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* factor_diff_matrix = factor_diff_.cpu_data();

    if (propagate_down[0]) {

        // compute diff for bottom    
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                inflate_backward(bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, *factor_);
            }
        }
    }
    
    // compute diff for factor_
    if (iter_counter_ >= this->layer_param().inflation_factor_param().start_iter()) {

	if (iter_counter_ % 4 == 0)    *factor_diff = 0;

        // dL/d(factor) = sum(top.diff[i,j] * d(top.data[i,j])/d(factor))
        Dtype sum_dLoss_dfactor = caffe_cpu_dot(top[0]->count(), factor_diff_matrix,  top_diff);
        
        const Dtype* top_factor_diff = top[1]->cpu_diff();       
        
	Dtype tmp = static_cast<Dtype>(1.0 * sum_dLoss_dfactor / num / height / width + top_factor_diff[0]);

        *factor_diff += tmp;
	if (this->layer_param().inflation_factor_param().clip_gradient() == true)
	{
	    float MARGIN = this->layer_param().inflation_factor_param().clip_gradient_value();
	    if (*factor_diff > MARGIN) *factor_diff = MARGIN;
	    if (*factor_diff < -MARGIN) *factor_diff = -MARGIN;
	}

        LOG(INFO) << " No." << iter_counter_ % 4
                  << "  factor: " << *factor_
                  << "  (" << height << " -> " << top_height << ")"
                  << "  f_diff: " << sum_dLoss_dfactor / num / height / width
		  << "  diff: " << tmp
		  << " Total diff: " << *factor_diff;
	if (iter_counter_ % 4 == 3)     LOG(INFO) << " Total diff: " << *factor_diff;

    } else {
        *factor_diff = 0;
        
        if (iter_counter_ == this->layer_param().inflation_factor_param().start_iter() && propagate_down[0])
            LOG(INFO) << " Start learning factor value ~~~~~";    
    }    
    iter_counter_++;         
}

#ifdef CPU_ONLY
STUB_GPU(InflationLayer);
#endif
INSTANTIATE_CLASS(InflationLayer);
REGISTER_LAYER_CLASS(Inflation);
}  // namespace caffe
