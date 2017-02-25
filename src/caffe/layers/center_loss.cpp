/*
* Modified center loss layer for segmentation.
* Author: Wei Zhen @ IIE, CAS
* Create on: 2016-12-25
* Last Modified: 2016-02-25
*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/center_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CenterLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  this->late_iter_ = this->layer_param_.center_loss_param().late_iter();  
  label_num_ = this->layer_param_.center_loss_param().label_num();  
  label_axis_ = bottom[0]->CanonicalAxisIndex(this->layer_param_.center_loss_param().axis());
  outer_num_ = bottom[0]->count(0, label_axis_);
  inner_num_ = bottom[0]->count(label_axis_+1);
  // Check if we need to set up the weights
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
        this->layer_param_.center_loss_param().center_filler()));
    center_filler->Fill(this->blobs_[0].get());

  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  // init label_counter_
  vector<int> label_counter_shape(1);
  label_counter_shape[0] = label_num_;
  label_counter_.Reshape(label_counter_shape);
  // init ignore labels
  for (int c = 0; c < this->layer_param_.center_loss_param().ignore_label_size(); ++c){
    ignore_label_.push_back(this->layer_param_.center_loss_param().ignore_label(c));
  }
  // init center inter distances
  center_mutual_distance.ReshapeLike(*this->blobs_[0]);
  vector<int> center_mutual_distance_dot_shape(1);
  // init iter counter
  count_ = 0;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[1]->channels(), 1);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  distance_.Reshape(bottom[0]->shape());
  variation_sum_.Reshape(this->blobs_[0]->shape());
  // bottom and label may not be in the same size, but have same ratio
  label_bottom_factor = bottom[1]->height() / float(bottom[0]->height());

  count_ ++;
}

// convert index on bottom[0] (data) onto bottom[1] (label)
template <typename Dtype>
inline int CenterLossLayer<Dtype>::label_idx_converter(int num, int label_height, int label_width, int data_idx, int data_width) {
  int y = int(data_idx / data_width);
  int x = data_idx % data_width;
  int y_ = int(y * label_bottom_factor);
  int x_ = int(x * label_bottom_factor);
  return num*label_height*label_width + y_*label_width + x_;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
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
  // center diff
  Dtype* variation_sum_data = variation_sum_.mutable_cpu_data();
  int* label_counter__ = label_counter_.mutable_cpu_data();
  
/*
  // find shortest center distance for each center
  Dtype tmp_distance = 1e20;
  Dtype* tmp_sub = (Dtype*)malloc(dim*sizeof(Dtype));
  Dtype* distance_inter = center_mutual_distance.mutable_cpu_data();
  const Dtype* center = this->blobs_[0]->cpu_data();
  for (int i = 0; i < label_num_; ++i) {
    if (find(ignore_label_.begin(), ignore_label_.end(), i) != ignore_label_.end())  continue;
    for (int j = 0; j < label_num_; ++j) {
	if (find(ignore_label_.begin(), ignore_label_.end(), j) != ignore_label_.end())  continue;
	if (i == j)  continue;
	// |current center (i) - another center (j)|^2, i != j
	caffe_sub(dim, center+i*dim, center+j*dim, tmp_sub);
	Dtype tmp = caffe_cpu_dot(dim, tmp_sub, tmp_sub);
	if (tmp < tmp_distance) {
	    tmp_distance = tmp;
	    caffe_copy(dim, tmp_sub, distance_inter+i*dim);
	}
    }
  }
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
	// |current center (i) - another center (j)|^2, i != j
	caffe_sub(dim, center+i*dim, center+j*dim, tmp_sub);
	caffe_axpy(dim, (Dtype)1./label_num_, tmp_sub, distance_inter+i*dim);
    }
  }

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
		// variation_sum_data(Y(n,1,y,x), c) -= D(n,c,x,y) + 2x center_mutual_distance
		variation_sum_data[label_value*dim + c] -= distance_data[c_idx*inner_num_ + j];
		//if (count_ > this->late_iter_)
		    //variation_sum_data[label_value*dim + c] += 2* *(center_mutual_distance.cpu_data()+label_value*dim+c);
		label_counter__[label_value]++;
	    }
        }
    }
  }
  // compute loss
  Dtype dot = caffe_cpu_dot(outer_num_ * inner_num_, distance_.cpu_data(), distance_.cpu_data());
  Dtype loss = dot / num / Dtype(2);
  top[0]->mutable_cpu_data()[0] = loss;

  delete [] tmp_sub;
}

template <typename Dtype>
void CenterLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  // Gradient with respect to centers
  if (this->param_propagate_down_[0]) {
    Dtype* center_diff = this->blobs_[0]->mutable_cpu_diff();
    const Dtype* variation_sum_data = variation_sum_.cpu_data();
    const int* label_counter__ = label_counter_.cpu_data();
    const int dim = bottom[0]->channels();

    // center's diff from other centers, update after late_iter_
    // second param in caffe_axpy is the control weight between the two diffs
    if (count_ > this->late_iter_) {
	caffe_set(this->blobs_[0]->count(), (Dtype)0., center_diff);
	caffe_axpy(dim*label_num_, (Dtype)0.1, center_mutual_distance.cpu_data(), center_diff);
	if (count_ == this->late_iter_+1)
	    LOG(INFO) << "Start computing mutual center diff.";
    }
    // center's diff from the cluster itself
    for (int label_value = 0; label_value < label_num_; label_value++) {
      // ignore label
      if (find(ignore_label_.begin(), ignore_label_.end(), label_value) != ignore_label_.end())  continue;
      caffe_axpy(dim, (Dtype)1./(label_counter__[label_value] + (Dtype)1.), variation_sum_data + label_value*dim, center_diff + label_value*dim);
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

  // Gradient with respect to bottom data 
  if (propagate_down[0] && count_ > this->late_iter_) {
    caffe_copy(distance_.count(), distance_.cpu_data(), bottom[0]->mutable_cpu_diff());
    caffe_scal(distance_.count(), top[0]->cpu_diff()[0] / bottom[0]->num(), bottom[0]->mutable_cpu_diff());
    if (count_ == this->late_iter_+1)
      LOG(INFO) << "Start backpropagating data gradient.";
  }
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
}

#ifdef CPU_ONLY
STUB_GPU(CenterLossLayer);
#endif

INSTANTIATE_CLASS(CenterLossLayer);
REGISTER_LAYER_CLASS(CenterLoss);

}  // namespace caffe

