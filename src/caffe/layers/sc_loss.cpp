/*
* Statistical contextual loss layer for segmentation.
* Author: Wei Zhen @ CS, HUST
* Create on: 2017-10-24
* Last Modified: 2017-10-24
*/

#include <vector>
#include <algorithm>

#include "caffe/filler.hpp"
#include "caffe/layers/sc_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void StatisticContextualLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->lambda_ = this->layer_param_.sc_loss_param().lambda(); 
  this->late_iter_ = this->layer_param_.sc_loss_param().late_iter(); 
  this->is_hard_aware_ = this->layer_param_.sc_loss_param().is_hard_aware();
  this->ld_margin_ = this->layer_param_.sc_loss_param().ld_margin();
  this->label_num_ = this->layer_param_.sc_loss_param().label_num();  
  this->center_decay_ = this->layer_param_.sc_loss_param().center_decay();
  // keep center decay align with original weight decay
  this->center_decay_ = this->layer_param_.param(0).lr_mult() == 0 ? 0 : this->center_decay_ / this->layer_param_.param(0).lr_mult();
  this->label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.sc_loss_param().axis());
  this->outer_num_ = bottom[0]->count(0, label_axis_);
  this->inner_num_ = bottom[0]->count(label_axis_+1);
  /*
   * 1. Check if we need to set up the weights
  */
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(1);
    // Intialize the weight
    vector<int> center_shape(2);
    // center feature shape: [label_num_, channel]
    center_shape[0] = label_num_;
    center_shape[1] = bottom[0]->channels();
    // this->blobs_ : treat center feature as parameter
    this->blobs_[0].reset(new Blob<Dtype>(center_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > center_filler(GetFiller<Dtype>(
        this->layer_param_.sc_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  
  /*
  * 2. init var
  */
  // init label_counter_
  vector<int> label_counter_shape(1);
  label_counter_shape[0] = label_num_;
  label_counter_.Reshape(label_counter_shape);
  // init ignore labels
  for (int c = 0; c < this->layer_param_.sc_loss_param().ignore_label_size(); ++c){
    ignore_label_.push_back(this->layer_param_.sc_loss_param().ignore_label(c));
  }
  // init center inter distances
  center_mutual_distance.ReshapeLike(*this->blobs_[0]);
  vector<int> center_mutual_distance_dot_shape(1);
  // init iter counter
  count_ = 0;

  // init of hard aware mode variables
  if (bottom.size() == 3 && this->is_hard_aware_ == true) {
    if (top.size() == 2)
      top[1]->Reshape(bottom[2]->shape());
    hw_sum = new Dtype[this->label_num_];
    hw_count = new Dtype[this->label_num_];
  }

  /*
  * 3. check center decay settings
  */
  // check weight decay is not used, use center decay only
  CHECK_EQ(this->layer_param_.param(0).decay_mult(), 0) << "Weight decay should not be used, use center decay only. decay_mult = "
						 << this->layer_param_.param(0).decay_mult() << "vs. 0.";
}

template <typename Dtype>
void StatisticContextualLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  /*
  * 1. init var
  */
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  distance_.Reshape(bottom[0]->shape());
  variation_sum_.Reshape(this->blobs_[0]->shape());
  // bottom and label may not be in the same size, but have same ratio
  label_bottom_factor = bottom[1]->height() / float(bottom[0]->height());

  /*
  * 2. generate hard aware flags: whether a pixel should be used to build cluster centers
  */
  // hard pix thred: mean loss of a label
  if (bottom.size() == 3 && this->is_hard_aware_ == true) {
    CHECK(bottom[0]->num() == bottom[2]->num() && bottom[0]->height() == bottom[2]->height() && bottom[0]->width() == bottom[2]->width())
	 << "Data input should have the same batch size, height and width with loss heat map input: "
	 << "("<< bottom[0]->shape()[0] << ","<< bottom[0]->shape()[1] << ","<< bottom[0]->shape()[2] << ","<< bottom[0]->shape()[3] << ") vs. ("
	 << bottom[2]->shape()[0] << ","<< bottom[2]->shape()[1] << ","<< bottom[2]->shape()[2] << ","<< bottom[2]->shape()[3] << ")";
    this->hard_aware_flags_.Reshape(bottom[2]->shape());
    // compute metirc for each label
    memset(hw_sum, 0, sizeof(Dtype)*this->label_num_);
    memset(hw_count, 1e-5, sizeof(Dtype)*this->label_num_);
    const Dtype* dense_loss = bottom[2]->cpu_data();
    const int label_height = bottom[1]->height();
    const int label_width = bottom[1]->width();
    const int data_width = bottom[0]->width();
    const Dtype* bottom_label = bottom[1]->cpu_data();
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int j = 0; j < this->inner_num_; ++j) {
	const int label_val = bottom_label[ label_idx_converter(n, label_height, label_width, j, data_width) ];
	hw_sum[label_val] += dense_loss[n*this->inner_num_ + j];
	hw_count[label_val]++;
      }
    }
    // set hard aware flags
    Dtype* hw_flags = this->hard_aware_flags_.mutable_cpu_data();
    caffe_div(this->label_num_, hw_sum, hw_count, hw_sum);  // for mean metric
    for (int n = 0; n < bottom[0]->num(); ++n) {
      for (int j = 0; j < this->inner_num_; ++j) {
	const int label_val = bottom_label[ label_idx_converter(n, label_height, label_width, j, data_width) ];
	if (dense_loss[n*this->inner_num_ + j] <= hw_sum[label_val])
	  hw_flags[n*this->inner_num_ + j] = 1;	//>0 true
	else
	  hw_flags[n*this->inner_num_ + j] = -1;//<0 false
      }
    }
    if (top.size()==2)
      caffe_copy(this->hard_aware_flags_.count(), this->hard_aware_flags_.cpu_data(), top[1]->mutable_cpu_data());
  }// end of generating hard aware flags

  count_ ++;
}

// convert index on bottom[0] (data) onto bottom[1] (label)
template <typename Dtype>
inline int StatisticContextualLossLayer<Dtype>::label_idx_converter(int num, int label_height, int label_width, int data_idx, int data_width) {
  int y = int(data_idx / data_width);
  int x = data_idx % data_width;
  int y_ = int(y * label_bottom_factor);
  int x_ = int(x * label_bottom_factor);
  return num*label_height*label_width + y_*label_width + x_;
}

template <typename Dtype>
void StatisticContextualLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  const int num = bottom[0]->num();
  const int dim = bottom[0]->channels();
  const Dtype* center = this->blobs_[0]->cpu_data();
  Dtype* distance_data = distance_.mutable_cpu_data();
  const int label_height = bottom[1]->height();
  const int label_width = bottom[1]->width();
  const int data_width = bottom[0]->width();
  const Dtype* hw_flags = this->hard_aware_flags_.cpu_data();
  // center diff
  Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
  int* label_counter__ = label_counter_.mutable_cpu_data();
  
  /*
  * 1. inter center distances
  */
  // accumulate all mutual center distances
  Dtype* tmp_sub = (Dtype*)malloc(dim*sizeof(Dtype));
  Dtype* distance_inter = center_mutual_distance.mutable_cpu_data();
  // reset mutual center distances
  caffe_set(center_mutual_distance.count(), (Dtype)0., distance_inter);
  for (int i = 0; i < label_num_; ++i) {
    if (find(ignore_label_.begin(), ignore_label_.end(), i) != ignore_label_.end())  continue;
    for (int j = 0; j < label_num_; ++j) {
	if (find(ignore_label_.begin(), ignore_label_.end(), j) != ignore_label_.end())  continue;
	if (i == j)  continue;
	// |current center (i,i) - another center (j,i)|^2, i != j
	// |current center (i,j) - another center (j,j)|^2, i != j
	tmp_sub[i] = center[i*dim + i] - center[j*dim + i];
	tmp_sub[j] = center[i*dim + j] - center[j*dim + j];
	caffe_add(dim, tmp_sub, distance_inter+i*dim, distance_inter+i*dim);
    }
    distance_inter[i*dim + i] /= (label_num_ - ignore_label_.size());
    // L_{D} = max(ld_margin_ - center_mutual_distance^2, 0)
    if (caffe_cpu_dot(dim, distance_inter+i*dim, distance_inter+i*dim) > this->ld_margin_)
      caffe_set(dim, (Dtype)0., distance_inter+i*dim);
  }

  /*
  * 2. class intra distances and intra term diffs
  */
  // the i-th distance_data
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < dim; ++c) {
        for (int j = 0; j < inner_num_; ++j) {
	    const int label_idx = label_idx_converter(n, label_height, label_width, j, data_width);
            const int label_value = static_cast<int>(bottom_label[label_idx]);
	    // ignore label
	    if (find(ignore_label_.begin(), ignore_label_.end(), label_value) != ignore_label_.end())  continue;
            // dC(n,c,y,x) = X(n,c,y,x) - C(c,Y(n,1,y,x))
            const int c_idx = n*dim+c;
            distance_data[c_idx*inner_num_ + j] = bottom_data[c_idx*inner_num_ + j] - center[label_value*dim + c];
	    // compute center diff
	    if (this->param_propagate_down_[0]) {
		// variation_sum_data(Y(n,1,y,x), c) -= D(n,c,x,y) + 2x center_mutual_distance, center_mutual_distance is processed in bp func
		if (bottom.size() == 3 && this->is_hard_aware_ == true) {	// hard awared mode
		  if (hw_flags[n*inner_num_ + j] > 0) {				// flag>0 => true; flag<0 => false
		    variation_sum_data[label_value*dim + c] -= distance_data[c_idx*inner_num_ + j];
		    label_counter__[label_value]++;
		  }
		  else continue;
		}
		else {								// original mode
		  variation_sum_data[label_value*dim + c] -= distance_data[c_idx*inner_num_ + j];
		  label_counter__[label_value]++;
		}
	    }
        }
    }
  }
  for (int i = 0; i < label_counter_.count(); i++)
    label_counter__[i] /= dim;

  /*
  * 3. inter diffs
  */
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < dim; ++c) {
        for (int j = 0; j < inner_num_; ++j) {
	  const int label_idx = label_idx_converter(n, label_height, label_width, j, data_width);
          const int label_value = static_cast<int>(bottom_label[label_idx]);
	  // ignore label
	  if (find(ignore_label_.begin(), ignore_label_.end(), label_value) != ignore_label_.end())  continue;
          const int c_idx = n*dim+c;
	  // if don't use hard aware mode or use hard aware mode and the flag>0,
	  //    then propagate down inter loss diff
	  if (this->is_hard_aware_==false || (this->is_hard_aware_==true && hw_flags[n*inner_num_ + j]>0)) {
	    distance_data[c_idx*inner_num_ + j] += 2 * this->lambda_ / label_counter__[label_value] * distance_inter[label_value*dim+c];
	  }
	  // else, if use hard aware mode and the flag<=0, then do nothing
   	  else continue;
        }
    }
  }

  /*
  * 4. compute loss
  */
  Dtype dot = caffe_cpu_dot(outer_num_ * inner_num_, distance_.cpu_data(), distance_.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  delete [] tmp_sub;
}

template <typename Dtype>
void StatisticContextualLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  /*
  * 1. Gradient with respect to centers
  */
  if (this->param_propagate_down_[0]) {
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* center = this->blobs_[0]->mutable_cpu_data();
    const Dtype* variation_sum_data = variation_sum_.cpu_data();
    const int* label_counter__ = label_counter_.cpu_data();
    const int dim = bottom[0]->channels();

    caffe_set(this->blobs_[0]->count(), (Dtype)0., center_diff);
    /*
    * 1.1 center's diff from other centers
    */
    // lambda is the control weight between the two diffs
    caffe_axpy(dim*label_num_, this->lambda_*2, center_mutual_distance.cpu_data(), center_diff);

    /*
    * 1.2 center's diff from the cluster itself
    */
    for (int label_value = 0; label_value < label_num_; label_value++) {
      // ignore label
      if (find(ignore_label_.begin(), ignore_label_.end(), label_value) != ignore_label_.end())  continue;
      caffe_axpy(dim, (Dtype)1./(label_counter__[label_value] + (Dtype)1.), variation_sum_data + label_value*dim, center_diff + label_value*dim);
      /*
      * 1.3 center decay
      */
      caffe_axpy(dim, (Dtype)this->center_decay_, center + label_value*dim, center_diff + label_value*dim);
      center_diff[label_value*dim + label_value] -= center[label_value*dim + label_value] * this->center_decay_;
    }

//Dtype a=0, b=0,c=0;
//for (int i = 0; i < dim; i++){
//a+=center_mutual_distance.cpu_data()[3*dim+i];
//b+=variation_sum_data[3*dim+i];
//c+=center_diff[3*dim+i];
//}
//printf("%f %f %f\n",a,b,c);

    // reset variation_sum_data
    caffe_set(variation_sum_.count(), (Dtype)0., variation_sum_.mutable_cpu_data());
    // reset label counter
    caffe_set(label_counter_.count(), (int)0., label_counter_.mutable_cpu_data());
  }

  /*
  * 2. Gradient with respect to bottom data (they are acutually computed in forward func)
  */
  if (propagate_down[0]) {
    caffe_copy(distance_.count(), distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    caffe_scal(distance_.count(), top[0]->cpu_diff()[0] / bottom[0]->num(), bottom[0]->mutable_cpu_diff());
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(StatisticContextualLossLayer);
#endif

INSTANTIATE_CLASS(StatisticContextualLossLayer);
REGISTER_LAYER_CLASS(StatisticContextualLoss);

}  // namespace caffe

