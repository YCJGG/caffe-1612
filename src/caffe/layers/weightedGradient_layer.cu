/*
# Give weights on gradient.
#
# Author: Wei Zhen @ IIE, CAS
# Create on: 2016-09-11
# Last modify: 2016-09-11
#
*/

#include <algorithm>
#include <vector>

#include "caffe/layer_factory.hpp"
#include "caffe/layers/weightedGradient_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void WeightedGradientLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   // do nothing in forward stage
   caffe_copy(top[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
}

template <typename Dtype>
__global__ void BackwardGPU(const int nthreads, const Dtype* label, Dtype* bottom_diff, const Dtype* top_diff, const int* label_set, int len_label_set, float weight)
{
    CUDA_KERNEL_LOOP(index, nthreads) {
	int i = 0;
	for(; i < len_label_set; i++)
	{
	    // label on this pixel is in label_set
	    // current gradient needs to be weighted
	    if(label_set[i] == static_cast<int>(label[index]))
	    {
		bottom_diff[index] = weight * top_diff[index];
		break;
	    }
	}
	// if not, copy gradient directly
	if(i==len_label_set)
	    bottom_diff[index] = top_diff[index];
    }
}

template <typename Dtype>
void WeightedGradientLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* label = bottom[1]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* top_diff = top[0]->gpu_diff();
    int map_size = bottom[0]->height() * bottom[0]->width();

    // convert vector into array
    int* a_label_set = (int*)malloc(this->label_set.size() * sizeof(int));
    for(int i = 0; i < this->label_set.size(); i++)
	a_label_set[i] = this->label_set[i];
    int* cu_label_set = NULL;
    cudaMalloc((void**)&cu_label_set, this->label_set.size() * sizeof(int));
    caffe_gpu_memcpy(this->label_set.size(), a_label_set, cu_label_set);

    // mulply a weight on a pixel's feature if its label is in label_set
    if (propagate_down[0] == true)
    {
	for(int n = 0; n < bottom[0]->num(); n++)
	{
	    for(int c = 0; c < bottom[0]->channels(); c++)
	    {
		BackwardGPU<Dtype><<<CAFFE_GET_BLOCKS(map_size), CAFFE_CUDA_NUM_THREADS>>>(map_size, label+n*map_size,
				bottom_diff+n*c*map_size, top_diff+n*c*map_size, cu_label_set, this->label_set.size(), this->weight);
	    }
	}
    }
}

INSTANTIATE_LAYER_GPU_FUNCS(WeightedGradientLayer);
}  // namespace caffe
