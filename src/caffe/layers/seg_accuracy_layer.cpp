#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/seg_accuracy_layer.hpp"


namespace caffe {

template <typename Dtype>
void SegAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  confusion_matrix_.clear();
  confusion_matrix_.resize(bottom[0]->channels());
  SegAccuracyParameter seg_accuracy_param = this->layer_param_.seg_accuracy_param();
  for (int c = 0; c < seg_accuracy_param.ignore_label_size(); ++c){
    ignore_label_.insert(seg_accuracy_param.ignore_label(c));
  }
  
  this->iter_ = 0;
  this->test_iter_ = seg_accuracy_param.test_iter();
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1)
    << "The label should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";
  //confusion_matrix_.clear(); 
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  int data_index, label_index;

  // remove old predictions if reset() flag is true
  if (this->iter_ == 0) {
    confusion_matrix_.clear();
  }

  for (int i = 0; i < num; ++i) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
	// Top-k accuracy
	std::vector<std::pair<Dtype, int> > bottom_data_vector;

	for (int c = 0; c < channels; ++c) {
	  data_index = (c * height + h) * width + w;
	  bottom_data_vector.push_back(std::make_pair(bottom_data[data_index], c));
	}

	std::partial_sort(
	  bottom_data_vector.begin(), bottom_data_vector.begin() + 1,
	  bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

	// check if true label is in top k predictions
	label_index = h * width + w;
	const int gt_label = static_cast<int>(bottom_label[label_index]);

	if (ignore_label_.count(gt_label) != 0) {
	  // ignore the pixel with this gt_label
	  continue;
	} else if (gt_label >= 0 && gt_label < channels) {
	  // current position is not "255", indicating ambiguous position
	  confusion_matrix_.accumulate(gt_label, bottom_data_vector[0].second);
	} else {
	  LOG(FATAL) << "Unexpected label " << gt_label << ". num: " << i 
              << ". row: " << h << ". col: " << w;
      	}
      }
    }
    // move to next channel
    bottom_data  += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
  }

  /* for debug
  LOG(INFO) << "confusion matrix info:" << confusion_matrix_.numRows() << "," << confusion_matrix_.numCols();
  confusion_matrix_.printCounts();
  */

  /*
	background -- 0, face -- 1, hair -- 2, nose -- 3, upper lip -- 4, mouth -- 5, lower lip -- 6, left eye -- 7, right eye -- 8, left blrow -- 9, right brow -- 10
  */
  /* for Helen
  if (this->iter_+1 == this->test_iter_){
    LOG(INFO) << "# F1 Score - background: "
              << (Dtype)confusion_matrix_.classF1(0, 0);
    LOG(INFO) << "# F1 Score - face: "
              << (Dtype)confusion_matrix_.classF1(1, 1);
    LOG(INFO) << "# F1 Score - nose: "
              << (Dtype)confusion_matrix_.classF1(3, 3);
    LOG(INFO) << "# F1 Score - upper lip: "
              << (Dtype)confusion_matrix_.classF1(4, 4);
    LOG(INFO) << "# F1 Score - mouth: "
              << (Dtype)confusion_matrix_.classF1(5, 5);
    LOG(INFO) << "# F1 Score - lower lip: "
              << (Dtype)confusion_matrix_.classF1(6, 6);
    LOG(INFO) << "# F1 Score - eye: "
              << (Dtype)confusion_matrix_.classF1(7, 8);
    LOG(INFO) << "# F1 Score - brows: "
              << (Dtype)confusion_matrix_.classF1(9, 10);
    LOG(INFO) << "# F1 Score - mouth all: "
              << (Dtype)confusion_matrix_.classF1(4, 6);
    LOG(INFO) << "# F1 Score - mean: "
              << (Dtype)confusion_matrix_.classF1(3, 10);
    LOG(INFO) << (Dtype)confusion_matrix_.classF1(0, 0) << ","
              << (Dtype)confusion_matrix_.classF1(1, 1) << ","
	      << (Dtype)confusion_matrix_.classF1(3, 3) << ","
	      << (Dtype)confusion_matrix_.classF1(4, 4) << ","
	      << (Dtype)confusion_matrix_.classF1(5, 5) << ","
	      << (Dtype)confusion_matrix_.classF1(6, 6) << ","
	      << (Dtype)confusion_matrix_.classF1(7, 8) << ","
	      << (Dtype)confusion_matrix_.classF1(9, 10) << ","
	      << (Dtype)confusion_matrix_.classF1(4, 6) << ","
	      << (Dtype)confusion_matrix_.classF1(3, 10);
    this->iter_ = 0;
  }   -end for Helen- */
  /* for LFW */
  if (this->iter_+1 == this->test_iter_){
    LOG(INFO) << "# F1 Score - background: "
              << (Dtype)confusion_matrix_.classF1(0, 0);
    LOG(INFO) << "# F1 Score - hair: "
              << (Dtype)confusion_matrix_.classF1(1, 1);
    LOG(INFO) << "# F1 Score - face skin: "
              << (Dtype)confusion_matrix_.classF1(2, 2);
    LOG(INFO) << "# F1 Score - mean: "
              << (Dtype)confusion_matrix_.classF1(1, 2);
    LOG(INFO) << (Dtype)confusion_matrix_.classF1(0, 0) << ","
              << (Dtype)confusion_matrix_.classF1(1, 1) << ","
	      << (Dtype)confusion_matrix_.classF1(2, 2) << ","
	      << (Dtype)confusion_matrix_.classF1(1, 2);
    this->iter_ = 0;
  }
  else this->iter_++;
}

INSTANTIATE_CLASS(SegAccuracyLayer);
REGISTER_LAYER_CLASS(SegAccuracy);

}  // namespace caffe
