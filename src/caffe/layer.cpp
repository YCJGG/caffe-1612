<<<<<<< HEAD
#include <boost/thread.hpp>
=======
>>>>>>> caffe-bvlc-dev/master
#include "caffe/layer.hpp"

namespace caffe {

<<<<<<< HEAD
template <typename Dtype>
void Layer<Dtype>::InitMutex() {
  forward_mutex_.reset(new boost::mutex());
}

template <typename Dtype>
void Layer<Dtype>::Lock() {
  if (IsShared()) {
    forward_mutex_->lock();
  }
}

template <typename Dtype>
void Layer<Dtype>::Unlock() {
  if (IsShared()) {
    forward_mutex_->unlock();
  }
}

=======
>>>>>>> caffe-bvlc-dev/master
INSTANTIATE_CLASS(Layer);

}  // namespace caffe
