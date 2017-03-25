#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/inflationX_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace caffe {

template <typename Dtype>
__global__ void InflateForwardGPU(const int nthreads,
          const Dtype* bottom_data, const int bottom_height, const int bottom_width, 
          Dtype *top_data, const int top_height, const int top_width, 
          const float factor, Dtype *factor_diff_matrix, const int margin,
          const float factor_bg_mask=1, const float bg_mask_weight=1, const Dtype* label=NULL) {
           
    const float anchor_y = 0; //(height - 1) / 2.0;
    const float anchor_x = 0; //(width - 1) / 2.0;
    
    const float normalizer = margin * margin * margin * margin;
          
    CUDA_KERNEL_LOOP(index, nthreads) {
        
        // index refers to to top_data
        const int y_t = index / top_width;
        const int x_t = index % top_width;
        
        // coordinate on target map
        const int idx_t = y_t * top_width + x_t;
            
        top_data[idx_t] = 0;
        factor_diff_matrix[idx_t] = 0;
            
        float y_s = y_t / factor;
        float x_s = x_t / factor;
            
        for (int n = MAX(floor(y_s - margin) + 1, 0); n < MIN(y_s + margin, bottom_height); n++) {
            for (int m = MAX(floor(x_s - margin) + 1, 0); m < MIN(x_s + margin, bottom_width); m++) {
             
                top_data[idx_t] += bottom_data[n * bottom_width + m] * (margin - abs(x_s - m)) * (margin - abs(y_s - n));
                    
                factor_diff_matrix[idx_t] += bottom_data[n * bottom_width + m] 
                                             * ((2 * (m >= x_s) - 1) * (margin - abs(y_s - n)) * (-(x_s - anchor_x) / factor)
                                               +(2 * (n >= y_s) - 1) * (margin - abs(x_s - m)) * (-(y_s - anchor_y) / factor));
		// when using background mask
		if (label!= NULL && label[int(round(idx_t*factor_bg_mask))] == 0)
		{
		    factor_diff_matrix[idx_t] *= bg_mask_weight;
		}
		if (label!= NULL && label[int(round(idx_t*factor_bg_mask))] == 255)
		{
		    factor_diff_matrix[idx_t] = 0;
		}
            }
        }
        // normalize
        top_data[idx_t] /= normalizer;
        factor_diff_matrix[idx_t] /= normalizer;
    }
}

template <typename Dtype>
void InflationXLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    Dtype* factor_;
    if (iter_counter_ >= this->layer_param().inflation_factor_param().start_iter())
    	factor_ = bottom[1]->mutable_cpu_data();
    else {
	factor_ = new Dtype();
	*factor_ = this->layer_param().inflation_factor_param().factor();
    }

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* factor_diff_matrix = factor_diff_.mutable_gpu_data();
    const Dtype* label = NULL;
    if (this->layer_param().inflation_factor_param().use_bg_mask() == true)
	label = bottom[1]->gpu_data();
    
    // new shape
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();

    // resize
    const int nthreads = top_height * top_width;
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channels; c++) {    
      
            const int index_in = (n * channels + c) * height * width;
            const int index_out = (n * channels + c) * top_height * top_width;

	    if (this->layer_param().inflation_factor_param().use_bg_mask() == true)
	    {
		const int index_label = n * bottom[1]->height() * bottom[1]->width();
                InflateForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix, margin_, this->factor_bg_mask, this->bg_mask_weight, label+index_label);
	    }
	    else
                InflateForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix, margin_);
        }
    }   
}

template <typename Dtype>
__global__ void InflateBackwardGPU(const int nthreads, 
            Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
            const Dtype *top_diff, const int top_height, const int top_width, 
            const float factor, const int margin) {

    const float normalizer = factor * factor * margin * margin * margin * margin;

    CUDA_KERNEL_LOOP(index, nthreads) {
        
        // index refers to to top_data
        const int n = index / bottom_width;
        const int m = index % bottom_width;
        
        const int idx_s = n * bottom_width + m;
        bottom_diff[idx_s] = 0;
        
        for (int y_t = MAX(floor((n - margin) * factor) + 1, 0); y_t < MIN((n + margin) * factor, top_height); y_t++) {
            for (int x_t = MAX(floor((m - margin) * factor) + 1, 0); x_t < MIN((m + margin) * factor, top_width); x_t++) {
                
                // diff
                bottom_diff[idx_s] += top_diff[y_t * top_width + x_t] 
                                      * (margin - abs((x_t / factor) - m)) * (margin - abs((y_t / factor) - n));
            }
        }
        bottom_diff[idx_s] /= normalizer;
    }
}

template <typename Dtype>
void InflationXLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();
    
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* factor_;
    if (iter_counter_ >= this->layer_param().inflation_factor_param().start_iter())
    	factor_ = bottom[1]->mutable_cpu_data();
    else {
	factor_ = new Dtype();
	*factor_ = this->layer_param().inflation_factor_param().factor();
    }
    Dtype* factor_diff = bottom[1]->mutable_cpu_diff();
    const Dtype* factor_diff_matrix = factor_diff_.cpu_data();

    if (propagate_down[0]) {

        // compute diff for bottom
        const int nthreads = height * width;
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                InflateBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, *factor_, margin_);
            }
        }
    }
    
    if (iter_counter_ >= this->layer_param().inflation_factor_param().start_iter()) {
    
	if (iter_counter_ % 20 == 0)    *factor_diff = 0;

        // compute diff for factor_
        // dL/d(factor) = sum(top.diff[i,j] * d(top.data[i,j])/d(factor))
        Dtype sum_dLoss_dfactor = caffe_cpu_dot(top[0]->count(), factor_diff_matrix, top[0]->cpu_diff());

        const Dtype* top_factor_diff = top[1]->cpu_diff();       
        
	Dtype tmp = static_cast<Dtype>(1.0 * sum_dLoss_dfactor / num / height / width + top_factor_diff[0]);
	tmp *= this->layer_param().inflation_factor_param().factor_diff_weight();

	if (this->layer_param().inflation_factor_param().clip_gradient() == true)
	{
	    float MARGIN = this->layer_param().inflation_factor_param().clip_gradient_value();
	    if (tmp > MARGIN) tmp = MARGIN;
	    if (tmp < -MARGIN) tmp = -MARGIN;
	}
        *factor_diff += tmp;

        LOG(INFO) << " No." << iter_counter_ % 20
                  << "  factor: " << *factor_
                  << "  (" << height << " -> " << top_height << ")"
                  << "  f_diff: " << sum_dLoss_dfactor / num / height / width
		  << "  diff: " << tmp
		  << " Total diff: " << *factor_diff;
	if (iter_counter_ % 20 == 19)    LOG(INFO) << " Total diff: " << *factor_diff;

    } else {
        *factor_diff = 0;
        
        if (iter_counter_ == this->layer_param().inflation_factor_param().start_iter() && propagate_down[0])
            LOG(INFO) << " Start learning factor value";    
    }
    iter_counter_++;
}

INSTANTIATE_LAYER_GPU_FUNCS(InflationXLayer);

}  // namespace caffe
