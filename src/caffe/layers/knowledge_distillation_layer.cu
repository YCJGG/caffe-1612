#include <algorithm>
#include <cfloat>
#include <vector>
#include <iostream>

#include "caffe/layers/knowledge_distillation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void Forward_with_ignore_label(const int nthreads,
		 const int outer_num_, const int inner_num_, const int c_num_, const int dim,
		 const Dtype* label, const int ignore_label_,
		 const Dtype* soft_label, const Dtype* prob_data,
		 Dtype* loss, Dtype* count) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index / c_num_ / inner_num_;
		int j = index / c_num_ % inner_num_;
		int c = index % c_num_;
        	const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        	if (label_value == ignore_label_) {
          		continue;
        	}   
       
		const int pos = i * dim + c * inner_num_ + j;
		loss[index] = -soft_label[pos] * (log(max(prob_data[pos], Dtype(FLT_MIN)))-log(max(soft_label[pos], Dtype(FLT_MIN))));
		count[i*inner_num_ + j] = 1;
	}
}

template <typename Dtype>
__global__ void Forward_without_ignore_label(const int nthreads,
		 const int outer_num_, const int inner_num_, const int c_num_, const int dim,
		 const Dtype* soft_label, const Dtype* prob_data,
		 Dtype* loss) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index / c_num_ / inner_num_;
		int j = index / c_num_ % inner_num_;
		int c = index % c_num_;
		const int pos = i * dim + c * inner_num_ + j;
		loss[index] = -soft_label[pos] * (log(max(prob_data[pos], Dtype(FLT_MIN)))-log(max(soft_label[pos], Dtype(FLT_MIN))));
	}
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // Both logits are divided by the temperature T.
  caffe_copy<Dtype>(bottom[0]->count(), bottom[0]->gpu_data(), s_logit_.mutable_gpu_data());
  caffe_gpu_scal(bottom[0]->count(), Dtype(1)/T, s_logit_.mutable_gpu_data());
  caffe_copy<Dtype>(bottom[1]->count(), bottom[1]->gpu_data(), t_logit_.mutable_gpu_data());
  caffe_gpu_scal(bottom[0]->count(), Dtype(1)/T, t_logit_.mutable_gpu_data());
  // The forward pass computes the softmax prob values.
  s_softmax_layer_->Forward(s_softmax_bottom_vec_, s_softmax_top_vec_);
  t_softmax_layer_->Forward(t_softmax_bottom_vec_, t_softmax_top_vec_);
  const Dtype* prob_data = s_prob_.cpu_data();
  const Dtype* soft_label = t_prob_.cpu_data();
  int dim = s_prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  // Compute KL divergence.
  if (bottom.size() == 3 && has_ignore_label_) { // label inputs and ignore_label are given.
    const Dtype* label = bottom[2]->gpu_data();
    const int c_num_ = bottom[0]->shape(softmax_axis_);
    const int nthreads = outer_num_ * inner_num_ * c_num_;

    Blob<Dtype> gpu_count;
    gpu_count.Reshape(1,1,1,outer_num_*inner_num_);
    Blob<Dtype> gpu_loss;
    gpu_loss.Reshape(1,1,1,outer_num_*inner_num_*c_num_);
    Blob<Dtype> gpu_adder;

    Forward_with_ignore_label<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
		(nthreads, outer_num_, inner_num_, c_num_, dim, label, ignore_label_, soft_label, prob_data, gpu_loss.mutable_gpu_data(), gpu_count.mutable_gpu_data());

    gpu_adder.Reshape(gpu_count.shape());
    caffe_set(gpu_adder.count(), Dtype(1), gpu_adder.mutable_cpu_data());
    count = static_cast<int>(caffe_cpu_dot(gpu_adder.count(), gpu_adder.cpu_data(), gpu_count.cpu_data()));

    gpu_adder.Reshape(gpu_loss.shape());
    caffe_set(gpu_adder.count(), Dtype(1), gpu_adder.mutable_cpu_data());
    loss = caffe_cpu_dot(gpu_adder.count(), gpu_adder.cpu_data(), gpu_loss.cpu_data());

  } else { // label inputs or ignore_label are not given.
    count = outer_num_ * inner_num_;

    const int c_num_ = bottom[0]->shape(softmax_axis_);
    const int nthreads = outer_num_ * inner_num_ * c_num_;

    Blob<Dtype> gpu_loss;
    gpu_loss.Reshape(1,1,1,outer_num_*inner_num_*c_num_);
    Blob<Dtype> gpu_adder;

    Forward_without_ignore_label<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
		(nthreads, outer_num_, inner_num_, c_num_, dim, soft_label, prob_data, gpu_loss.mutable_gpu_data());

    gpu_adder.Reshape(gpu_loss.shape());
    caffe_set(gpu_adder.count(), Dtype(1), gpu_adder.mutable_cpu_data());
    loss = caffe_cpu_dot(gpu_adder.count(), gpu_adder.cpu_data(), gpu_loss.cpu_data());
  }

  top[0]->mutable_cpu_data()[0] = loss / get_normalizer(normalization_, count);

  // filter out teacher's pred with high loss, kfxw@2017-11-02
  if (this->layer_param_.knowledge_distillation_param().filter_teacher_pred() == true && bottom.size() == 4) {
    for (int n = 0; n < bottom[0]->num(); n++) {
      Dtype* p_dense_loss = bottom[3]->mutable_cpu_data() + n*inner_num_;
      this->filtered_idx[n] = find_filtered_idx(p_dense_loss, inner_num_, this->layer_param_.knowledge_distillation_param().filter_top_k());
    }
  }
}

template <typename Dtype>
__global__ void Backward_with_ignore_label(const int nthreads,
		 const int outer_num_, const int inner_num_, const int c_num_, const int dim,
		 const Dtype* label, const int ignore_label_,
		 Dtype* bottom_diff, Dtype* count) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		int i = index / c_num_ / inner_num_;
		int j = index / c_num_ % inner_num_;
		int c = index % c_num_;
        	const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        	if (label_value == ignore_label_) {
          		bottom_diff[i * dim + c * inner_num_ + j] = 0;
        	} else {
			count[i*inner_num_ + j] = 1;
		}
	}
}

template <typename Dtype>
void KnowledgeDistillationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1] | (bottom.size() == 3 && propagate_down[2])) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to soft label nor label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const Dtype* prob_data = s_prob_.gpu_data();
    caffe_copy(s_prob_.count(), prob_data, bottom_diff);
    const Dtype* soft_label = t_prob_.gpu_data();
    int dim = s_prob_.count() / outer_num_;
    int count = outer_num_ * inner_num_;
    // The gradients here are multiplied by T,
    // which is T^2 (as suggested in the paper) * 1/T (logits divided by T).
    caffe_gpu_axpby<Dtype>(outer_num_*dim, -T, soft_label, T, bottom_diff);
    // If label inputs are given, set the gradients to 0 w.r.t. ignore_label.
    if (bottom.size() == 3 && has_ignore_label_) {
      const Dtype* label = bottom[2]->gpu_data();
      const int c_num_ = bottom[0]->shape(softmax_axis_);
      const int nthreads = outer_num_ * inner_num_ * c_num_;

      Blob<Dtype> gpu_count;
      gpu_count.Reshape(1,1,1,outer_num_*inner_num_);
      Blob<Dtype> gpu_adder;

      Backward_with_ignore_label<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>>
		(nthreads, outer_num_, inner_num_, c_num_, dim, label, ignore_label_, bottom_diff, gpu_count.mutable_gpu_data());

      gpu_adder.Reshape(gpu_count.shape());
      caffe_set(gpu_adder.count(), Dtype(1), gpu_adder.mutable_cpu_data());
      count = static_cast<int>(caffe_cpu_dot(gpu_adder.count(), gpu_adder.cpu_data(), gpu_count.cpu_data()));
    }

    // filter out teacher's pred with high loss, kfxw@2017-11-02
    if (this->layer_param_.knowledge_distillation_param().filter_teacher_pred() == true && bottom.size() == 4) {
      for (int n = 0; n < bottom[0]->num(); n++) {
        Dtype* bottom_diff_channel = bottom[0]->mutable_cpu_diff() + n*dim; 
        perform_filtering_on_diff(bottom_diff_channel, this->filtered_idx[n], bottom[0]->channels(), inner_num_);
      }
      // Note: sync bottom[0]'s cpu and gpu memory
      bottom_diff = bottom[0]->mutable_gpu_diff();
    }

    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] /
                        get_normalizer(normalization_, count);
    caffe_gpu_scal(s_prob_.count(), loss_weight, bottom_diff);
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(KnowledgeDistillationLayer);

}  // namespace caffe
