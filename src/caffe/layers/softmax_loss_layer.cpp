#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }

  // for BOOTSTRAPPING, init bootstrap idx vector - kfxw@2017-09-26
  if (this->layer_param_.loss_param().bootstrapping()) {
    this->unbootstrapped_idx = vector<vector<int> >(bottom[0]->num());
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
  outer_num_ = bottom[0]->count(0, softmax_axis_);
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // original: softmax output
    // now: output dense loss heat map -- my-dev feature
    top[1]->ReshapeLike(*bottom[1]);
  }
}

template <typename Dtype>
Dtype SoftmaxWithLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = Dtype(valid_count);
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    // add new normalizer: BOOTSTRAPPING - kfxw@2017-09-26
    // equals to 'top K' numbers
    case LossParameter_NormalizationMode_BOOTSTRAP:
      if (!this->layer_param_.loss_param().bootstrapping())
	LOG(FATAL) << "Normalization Mode BOOTSTRAP must be used with bootstrapping softmax loss.";
      normalizer = Dtype(outer_num_ * this->layer_param_.loss_param().bootstrapping_top_k());
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}

// used for bootstrapping - kfxw@2017-09-26
template <typename Dtype>
vector<int> SoftmaxWithLossLayer<Dtype>::find_unbootstrapped_idx(Dtype* dense_loss, int array_size, int topK) {
	// perform argsort, descending
	vector<size_t> idx(array_size);
	std::iota(idx.begin(), idx.end(), 0);
	std::sort(idx.begin(), idx.end(),
		 [&dense_loss](size_t i1, size_t i2) {return dense_loss[i1] > dense_loss[i2];});	// lambda function
	// take first K idx
	vector<int> res(array_size-topK);
	std::copy(idx.begin()+topK, idx.end(), res.begin());
	return res;
}

// used for bootstrapping - kfxw@2017-09-26
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::perform_bootstrap_on_loss(Dtype* dense_loss, vector<int>un_boot_idx) {
	for (vector<int>::size_type i = 0; i != un_boot_idx.size(); ++i) {
		dense_loss[un_boot_idx[i]] = 0;
	}
}

// used for bootstrapping - kfxw@2017-09-26
template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::perform_bootstrap_on_diff(Dtype* diff, vector<int>un_boot_idx, int label_num, int inner_num) {
	for (vector<int>::size_type i = 0; i != un_boot_idx.size(); ++i) {
		for (int j = 0; j < label_num; j++) {
			diff[j*inner_num + un_boot_idx[i]] = 0;
		}
	}
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Blob<Dtype> dense_loss;	// dense loss heat map, the same size as label - kfxw@2017-02-23
  dense_loss.ReshapeLike(*bottom[1]);
  Dtype* p_dense_loss = dense_loss.mutable_cpu_data();
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));
      p_dense_loss[i * inner_num_ + j] = -log(std::max(prob_data[i * dim + label_value * inner_num_ + j],
                           Dtype(FLT_MIN)));
      ++count;
    }
  }

  // output loss heat map
  if (top.size() == 2) {
    // original: top[1]->ShareData(prob_);
    // now: output dense loss heat map -- my-dev feature
    caffe_copy(dense_loss.count(), p_dense_loss, top[1]->mutable_cpu_data());
  }

  // if use BOOTSTRAPPING mode - kfxw@2017-09-26
  // 1. sort dense loss map and find pix idx with top K losses
  if (this->layer_param_.loss_param().bootstrapping()) {
    for (int n = 0; n < bottom[0]->num(); n++) {
      p_dense_loss = dense_loss.mutable_cpu_data() + n*inner_num_;
      unbootstrapped_idx[n] = this->find_unbootstrapped_idx(p_dense_loss, inner_num_, this->layer_param_.loss_param().bootstrapping_top_k());
      // 2. modify dense loss map: reserve loss value on the K idx and set to 0 on other pix
      this->perform_bootstrap_on_loss(p_dense_loss, unbootstrapped_idx[n]);
    }
  }

  // loss normalization
  Dtype normalizer = get_normalizer(normalization_, count);
  caffe_scal<Dtype>(dense_loss.count(), 1.0/normalizer, p_dense_loss);
  Dtype loss = caffe_cpu_asum<Dtype>(dense_loss.count(), p_dense_loss);
  top[0]->mutable_cpu_data()[0] = loss / normalizer;
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    int count = 0;
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          ++count;
        }
      }
      if (this->layer_param_.loss_param().bootstrapping()) {
  	// if use BOOTSTRAPPING mode - kfxw@2017-09-26
  	// modify bottom_diff: reserve diff value on the K idx and set to 0 on other pix
        this->perform_bootstrap_on_diff(bottom_diff+i*dim, unbootstrapped_idx[i], bottom[0]->channels(), inner_num_);
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SoftmaxWithLoss);

}  // namespace caffe
