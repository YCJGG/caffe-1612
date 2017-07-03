#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/inflation_layer.hpp"
#include "caffe/util/math_functions.hpp"

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

namespace caffe {

template <typename Dtype>
__global__ void InflateForwardGPU(const int nthreads,
          const Dtype* bottom_data, const int bottom_height, const int bottom_width, 
          Dtype *top_data, const int top_height, const int top_width, 
          const float factor, Dtype *factor_diff_matrix,
          const float factor_bg_mask=1, const float bg_mask_weight=1, const Dtype* label=NULL) {
           
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
            
        for (int n = MAX(floor(y_s - 1) + 1, 0); n < MIN(y_s + 1, bottom_height); n++) {
            for (int m = MAX(floor(x_s - 1) + 1, 0); m < MIN(x_s + 1, bottom_width); m++) {
             
                top_data[idx_t] += bottom_data[n * bottom_width + m] * (1 - abs(x_s - m)) * (1 - abs(y_s - n));
                    
                factor_diff_matrix[idx_t] += bottom_data[n * bottom_width + m] 
                                             * ((2 * (m >= x_s) - 1) * (1 - abs(y_s - n)) * (-x_s / factor)
                                               +(2 * (n >= y_s) - 1) * (1 - abs(x_s - m)) * (-y_s / factor));
		// when using background mask
		if (label!= NULL && label[int(round(idx_t*factor_bg_mask))] == 0)
		{
		    factor_diff_matrix[idx_t] *= bg_mask_weight;
		}
            }
        }
    }
}

template <typename Dtype>
void InflationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const Dtype* factor_ = this->blobs_[0]->cpu_data();

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
                InflateForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix, this->factor_bg_mask, this->bg_mask_weight, label+index_label);
	    }
	    else
                InflateForwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_data + index_in, height, width, top_data + index_out, top_height, top_width, *factor_, factor_diff_matrix);
        }
    }   
}

template <typename Dtype>
__global__ void InflateBackwardGPU(const int nthreads, 
            Dtype *bottom_diff, const int bottom_height, const int bottom_width, 
            const Dtype *top_diff, const int top_height, const int top_width, 
            const float factor) {

    const float normalizer = factor * factor;

    CUDA_KERNEL_LOOP(index, nthreads) {
        
        // index refers to to top_data
        const int n = index / bottom_width;
        const int m = index % bottom_width;
        
        const int idx_s = n * bottom_width + m;
        bottom_diff[idx_s] = 0;
        
        for (int y_t = MAX(floor((n - 1) * factor) + 1, 0); y_t < MIN((n + 1) * factor, top_height); y_t++) {
            for (int x_t = MAX(floor((m - 1) * factor) + 1, 0); x_t < MIN((m + 1) * factor, top_width); x_t++) {
                
                // diff
                bottom_diff[idx_s] += top_diff[y_t * top_width + x_t] 
                                      * (1 - abs((x_t / factor) - m)) * (1 - abs((y_t / factor) - n));
            }
        }
        bottom_diff[idx_s] /= normalizer;
    }
}

template <typename Dtype>
void InflationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    

    // get parameters
    const int num = bottom[0]->num();
    const int channels = bottom[0]->channels();
    const int height = bottom[0]->height();
    const int width = bottom[0]->width();
    const int top_height = top[0]->height();
    const int top_width = top[0]->width();
    
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* factor_ = this->blobs_[0]->cpu_data();
    Dtype* factor_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* factor_diff_matrix = factor_diff_.cpu_data();

    if (propagate_down[0]) {

        // compute diff for bottom
        const int nthreads = height * width;
        
        for (int n = 0; n < num; n++) {
            for (int c = 0; c < channels; c++) {
                const int index_in = (n * channels + c) * height * width;
                const int index_out = (n * channels + c) * top_height * top_width;
                InflateBackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>(nthreads, bottom_diff + index_in, height, width, top_diff + index_out, top_height, top_width, *factor_);
            }
        }
    }
    
    if (iter_counter_ >= this->layer_param().inflation_factor_param().start_iter()) {
    
	if (iter_counter_ % 4 == 0)    *factor_diff = 0;

        // compute diff for factor_
        // dL/d(factor) = sum(top.diff[i,j] * d(top.data[i,j])/d(factor))
        Dtype sum_dLoss_dfactor = caffe_cpu_dot(top[0]->count(), factor_diff_matrix,  top_diff);
        
        const Dtype* top_factor_diff = top[1]->cpu_diff();       
        
	Dtype tmp = static_cast<Dtype>(1.0 * sum_dLoss_dfactor / num / height / width + top_factor_diff[0]);

	if (this->layer_param().inflation_factor_param().clip_gradient() == true)
	{
	    float MARGIN = this->layer_param().inflation_factor_param().clip_gradient_value();
	    tmp = max(min(tmp, MARGIN), -MARGIN);
	}
        *factor_diff += tmp;
	if (this->layer_param().inflation_factor_param().clip_gradient() == true)
	{
	    float MARGIN = this->layer_param().inflation_factor_param().clip_gradient_value();
	    *factor_diff = max(min(*factor_diff, MARGIN), -MARGIN);
	}

        LOG(INFO) << " No." << iter_counter_ % 4
                  << "  factor: " << *factor_
                  << "  (" << height << " -> " << top_height << ")"
                  << "  f_diff: " << sum_dLoss_dfactor / num / height / width
		  << "  diff: " << tmp
		  << " Total diff: " << *factor_diff;
	if (iter_counter_ % 4 == 3)    LOG(INFO) << " Total diff: " << *factor_diff;

    } else {
        *factor_diff = 0;
        
        if (iter_counter_ == this->layer_param().inflation_factor_param().start_iter() && propagate_down[0])
            LOG(INFO) << " Start learning factor value ~~~~~";          
    }  
    iter_counter_++;    
}

INSTANTIATE_LAYER_GPU_FUNCS(InflationLayer);

}  // namespace caffe
