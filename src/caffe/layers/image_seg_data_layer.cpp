#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/image_seg_data_layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/data_transformer.hpp"

namespace caffe {

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  BaseDataLayer<Dtype>::LayerSetUp(bottom, top);
  if (top.size() >= 3) {
    this->output_data_dim_ = true;
  } else {
    this->output_data_dim_ = false;
  }
  if (top.size() == 4) {
    this->output_additional_channel_ = true;
  } else {
    this->output_additional_channel_ = false;
  }
  // Now, start the prefetch thread. Before calling prefetch, we make two
  // cpu_data calls so that the prefetch thread does not accidentally make
  // simultaneous cudaMalloc calls when the main thread is running. In some
  // GPUs this seems to cause failures if we do not so.
  this->prefetch_data_.mutable_cpu_data();
  if (this->output_labels_) {
    this->prefetch_label_.mutable_cpu_data();
  }
  if (this->output_data_dim_) {
    this->prefetch_data_dim_.mutable_cpu_data();
  }
  if (this->output_additional_channel_) {
    this->prefetch_additional_channel_.mutable_cpu_data();
  }

  DLOG(INFO) << "Initializing prefetch";
  this->CreatePrefetchThread();
  DLOG(INFO) << "Prefetch initialized.";
}


template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // First, join the thread
  BasePrefetchingDataLayer<Dtype>::JoinPrefetchThread();
  DLOG(INFO) << "Thread joined";
  // Copy the data
  caffe_copy(this->prefetch_data_.count(), this->prefetch_data_.cpu_data(),
             top[0]->mutable_cpu_data());
  DLOG(INFO) << "Prefetch copied";
  if (this->output_labels_) {
    caffe_copy(this->prefetch_label_.count(), this->prefetch_label_.cpu_data(),
               top[1]->mutable_cpu_data());
  }
  if (this->output_data_dim_) {
    caffe_copy(this->prefetch_data_dim_.count(), this->prefetch_data_dim_.cpu_data(),
	       top[2]->mutable_cpu_data());
  }
  if (this->output_additional_channel_) {
    caffe_copy(this->prefetch_additional_channel_.count(), this->prefetch_additional_channel_.cpu_data(),
	       top[3]->mutable_cpu_data());
  }

  // Start a new prefetch thread
  DLOG(INFO) << "CreatePrefetchThread";
  BasePrefetchingDataLayer<Dtype>::CreatePrefetchThread();
}

#ifdef CPU_ONLY
STUB_GPU_FORWARD(ImageDimPrefetchingDataLayer, Forward);
#endif
INSTANTIATE_CLASS(ImageDimPrefetchingDataLayer);

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  if (top.size() >= 3) {
    this->output_data_dim_ = true;
  } else {
    this->output_data_dim_ = false;
  }
  if (top.size() == 4) {
    this->output_additional_channel_ = true;
  } else {
    this->output_additional_channel_ = false;
  }

  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  const int label_type = this->layer_param_.image_data_param().label_type();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());

  string linestr;
  while (std::getline(infile, linestr)) {
    std::istringstream iss(linestr);
    string imgfn;
    iss >> imgfn;
    string segfn = "";
    string addfn = "";
    if (label_type != 0) { //ImageDataParameter_LabelType_NONE = 0
      iss >> segfn;
    }
    if (this->output_additional_channel_) { //this->output_additional_channel_ == True
      iss >> addfn;
    }

    std::vector<string> line_;
    line_.push_back(imgfn);
    line_.push_back(segfn);
    line_.push_back(addfn);
    lines_.push_back(line_);
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_][0],
                                    new_height, new_width, is_color);
  const int channels = cv_img.channels();
  const int height = cv_img.rows;
  const int width = cv_img.cols;

  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  if (crop_size > 0) {
    // image
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);

    //label
    top[1]->Reshape(batch_size, 1, crop_size, crop_size);
    this->prefetch_label_.Reshape(batch_size, 1, crop_size, crop_size);
    this->transformed_label_.Reshape(1, 1, crop_size, crop_size);

    //additional channel
    if (this->output_additional_channel_) {
	top[3]->Reshape(batch_size, 1, crop_size, crop_size);
	this->prefetch_additional_channel_.Reshape(batch_size, 1, crop_size, crop_size);
	this->transformed_additional_channel_.Reshape(1, 1, crop_size, crop_size);
    }
     
  } else {
    // image
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    this->prefetch_label_.Reshape(batch_size, 1, height, width);
    this->transformed_label_.Reshape(1, 1, height, width);  

    //additional channel
    if (this->output_additional_channel_) {
	top[3]->Reshape(batch_size, 1, crop_size, crop_size);
	this->prefetch_additional_channel_.Reshape(batch_size, 1, height, width);
	this->transformed_additional_channel_.Reshape(1, 1, height, width);
    }   
  }

  // image dimensions, for each image, stores (img_height, img_width)
  if (this->output_data_dim_) {
    top[2]->Reshape(batch_size, 1, 1, 2);
    this->prefetch_data_dim_.Reshape(batch_size, 1, 1, 2);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
	    << top[0]->channels() << "," << top[0]->height() << ","
	    << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
	    << top[1]->channels() << "," << top[1]->height() << ","
	    << top[1]->width();
  // image_dim
  if (this->output_data_dim_) {
    LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
	      << top[2]->channels() << "," << top[2]->height() << ","
	      << top[2]->width();
  }
  // additional
  if (this->output_additional_channel_) {
    LOG(INFO) << "output additional_channel size: " << top[3]->num() << ","
	      << top[3]->channels() << "," << top[3]->height() << ","
	      << top[3]->width();
  }
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageSegDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());

  Dtype* top_data     	= this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label    	= this->prefetch_label_.mutable_cpu_data();
  Dtype* top_data_dim;
  Dtype* top_additional_channel;
  if (this->output_data_dim_) { 
    top_data_dim = this->prefetch_data_dim_.mutable_cpu_data();
  }
  if (this->output_additional_channel_) { 
    top_additional_channel = this->prefetch_additional_channel_.mutable_cpu_data();
  }

  const int max_height = this->prefetch_data_.height();
  const int max_width  = this->prefetch_data_.width();

  ImageDataParameter image_data_param    = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width  = image_data_param.new_width();
  const int label_type = this->layer_param_.image_data_param().label_type();
  const int ignore_label = image_data_param.ignore_label();
  const bool is_color  = image_data_param.is_color();
  string root_folder   = image_data_param.root_folder();

  const int lines_size = lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    std::vector<cv::Mat> cv_img_seg;

    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);

    // read image (top[0])
    int img_row, img_col;
    cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_][0],
	  new_height, new_width, is_color, &img_row, &img_col));
    if (!cv_img_seg[0].data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_][0];
    }

    // read seg label (top[1])
    if (label_type == ImageDataParameter_LabelType_PIXEL) {	// LabelType == PIXEL: load seg label from image
      cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_][1],
					    new_height, new_width, false));
      if (!cv_img_seg[1].data) {
	DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_][1];
      }
    }
    else if (label_type == ImageDataParameter_LabelType_IMAGE) {// LabelType == IMAGE: all labels are a scalar specified in infile
      const int label = atoi(lines_[lines_id_][1].c_str());
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, CV_8UC1, cv::Scalar(label));
      cv_img_seg.push_back(seg);      
    }
    else {							// No specified LabelType: all labels are ignore_labels
      cv::Mat seg(cv_img_seg[0].rows, cv_img_seg[0].cols, CV_8UC1, cv::Scalar(ignore_label));
      cv_img_seg.push_back(seg);
    }

    // read data_dim (top[2])
    if (this->output_data_dim_) {
      int top_data_dim_offset = this->prefetch_data_dim_.offset(item_id);
      top_data_dim[top_data_dim_offset]     = static_cast<Dtype>(std::min(max_height, img_row));
      top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));
    }

    // read additional channel (top[3])
    if (this->output_additional_channel_) {
      cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_][2],
					    new_height, new_width, false));
      if (!cv_img_seg[2].data) {
	DLOG(INFO) << "Fail to load additional channel: " << root_folder + lines_[lines_id_][2];
      }
    }

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;

    offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);

    offset = this->prefetch_label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);

    if (this->output_additional_channel_) {
      this->transformed_additional_channel_.set_cpu_data(top_additional_channel + offset);
      this->data_transformer_->TransformImgAndSegAndAddChannel(cv_img_seg, 
	   &(this->transformed_data_), &(this->transformed_label_), &(this->transformed_additional_channel_),
	   ignore_label);
    }
    else {
    this->data_transformer_->TransformImgAndSeg(cv_img_seg, 
	 &(this->transformed_data_), &(this->transformed_label_),
	 ignore_label);
    }

    trans_time += timer.MicroSeconds();

    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

#ifdef CPU_ONLY
STUB_GPU(ImageSegDataLayer);
#endif
INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(ImageSegData);
}  // namespace caffe
