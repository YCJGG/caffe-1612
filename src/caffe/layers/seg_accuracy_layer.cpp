#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <iostream>
#include <string.h>
#include <stdlib.h>

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
  
  // init iter counter
  this->iter_ = 0;
  this->test_iter_ = seg_accuracy_param.test_iter();

  // init plugin config file of different dataset
  memset(this->plugin_name, 0, sizeof(char)*1024);
  strcat(this->plugin_name, seg_accuracy_param.plugin_name().c_str());
  //strcat(this->plugin_name, ".wzconfig");
  CHECK(access(this->plugin_name,0) == 0) << "Seg Acc Layer config file "
			<< this->plugin_name << " doesn't exist. " << access(this->plugin_name,0);
  LOG(INFO) << "Using Seg Acc Layer config file " << this->plugin_name;

  // read plugin config file
  char buffer[1024];
  std::ifstream in(this->plugin_name);
  while(in.getline(buffer, 1024)) {
    // split by '&'
    char* split_buffer = strtok(buffer, "&");
    for (int i = 0; i < 3; i++) {
      CHECK(split_buffer != NULL) << "Plugin config file bad format, item " << i;
      // item example: print info & fromIndex & toIndex
      if(i==0) {
	string tmp(split_buffer);
	this->plugin_item_info.insert(this->plugin_item_info.end(), tmp);
      } else if(i==1)  this->plugin_item_from.insert(this->plugin_item_from.end(), atoi(split_buffer));
	else if(i==2)  this->plugin_item_to.insert(this->plugin_item_to.end(), atoi(split_buffer));
      split_buffer = strtok(NULL, "&"); // loop to next string split
    }
  }
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
              << ". row: " << h << ". col: " << w << ". at iter: " << this->iter_;
      	}
      }
    }
    // move to next channel
    bottom_data  += bottom[0]->offset(1);
    bottom_label += bottom[1]->offset(1);
  }

  if (this->iter_+1 == this->test_iter_) {
     vector<Dtype> scores;
     // evaluation and display
     switch(this->layer_param_.seg_accuracy_param().eval_metrc()) {
        case SegAccuracyParameter_EvalMetric_F1_SCORE:
	    for (int i = 0; i < this->plugin_item_info.size(); i++) {
		scores.insert(scores.end(),
		 	(Dtype)confusion_matrix_.classF1(this->plugin_item_from[i], this->plugin_item_to[i]));
		LOG(INFO) << this->plugin_item_info[i] << (Dtype)scores[i];
	    }
	    break;
        case SegAccuracyParameter_EvalMetric_JACCARD:
	    for (int i = 0; i < this->plugin_item_info.size()-1; i++) {
		// assume that no labels are combined/merged
		scores.insert(scores.end(),
		 	(Dtype)confusion_matrix_.jaccard(this->plugin_item_from[i]));//, this->plugin_item_to[i]));
		LOG(INFO) << this->plugin_item_info[i] << (Dtype)scores[i];
	    }
	    // mean Jaccard score
	    scores.insert(scores.end(), (Dtype)confusion_matrix_.avgJaccard());
	    LOG(INFO) << this->plugin_item_info[ this->plugin_item_info.size()-1 ] << (Dtype)scores[ this->plugin_item_info.size()-1 ];
	    break;
	default:
	{
		scores.insert(scores.end(),0);
		LOG(INFO) << "Warning: Unkonw evaluation metric";
	}
    }
    
    // display all results
    char log_buffer[4096];
    memset(log_buffer, 0, 1);
    for (int i = 0; i < scores.size(); i++) {
	char stmp[64];
	sprintf(stmp, "%.4lf,", scores[i]);
	strcat(log_buffer, stmp);
    }
    LOG(INFO) << log_buffer;

    this->iter_ = 0;
  }
  else this->iter_++;
}

INSTANTIATE_CLASS(SegAccuracyLayer);
REGISTER_LAYER_CLASS(SegAccuracy);

}  // namespace caffe
