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
void WeightedGradientLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    //input label map (bottom[1]) should have the same size as input feature map (bottom[0])
    CHECK_EQ(bottom[0]->height(), bottom[1]->height()) <<"input label map should have the same size as input feature map ("
						       << bottom[0]->height() << " vs. " << bottom[1]->height() << ")";
    //get weighted label set and weigt
    WeightedGradientParameter wg_param = this->layer_param().weighted_gradient_param();
    this->weight = wg_param.weight();
    this->label_set.clear();
    std::copy(wg_param.weighted_label().begin(), wg_param.weighted_label().end(), std::back_inserter(this->label_set));
}

template <typename Dtype>
void WeightedGradientLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
    top[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void WeightedGradientLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
   // do nothing in forward stage
   caffe_copy(top[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
}

template <typename Dtype>
void WeightedGradientLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* top_diff = top[0]->cpu_diff();
    int map_size = bottom[0]->height() * bottom[0]->width();

    // mulply a weight on a pixel's feature if its label is in label_set
    if (propagate_down[0] == true)
    {
	for(int n = 0; n  < bottom[0]->num(); n++)
	{
	    for(int c = 0; c < bottom[0]->channels(); c++)
	    {
		for(int inner = 0; inner < map_size; inner++)
		{
		    // label on this pixel is in label_set
		    // current gradient needs to be weighted
		    if ( std::find(this->label_set.begin(), this->label_set.end(), static_cast<int>(label[n*map_size+inner])) != this->label_set.end() )
		    {    bottom_diff[n*c*map_size + inner] = this->weight * top_diff[n*c*map_size + inner];	}
		    // if not, copy gradient directly
		    else
		    {    bottom_diff[n*c*map_size + inner] = top_diff[n*c*map_size + inner];     }
		}
	    }
	}
    }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedGradientLayer);
#endif
INSTANTIATE_CLASS(WeightedGradientLayer);
REGISTER_LAYER_CLASS(WeightedGradient);
}  // namespace caffe
