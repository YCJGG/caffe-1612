<<<<<<< HEAD
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif
#include <glog/logging.h>
#include <stdio.h>

=======
#ifdef USE_NCCL

#include <cuda_runtime.h>
#include <glog/logging.h>
#include <stdio.h>
>>>>>>> caffe-bvlc-dev/master
#include <sstream>
#include <string>
#include <vector>

<<<<<<< HEAD
#include "boost/thread.hpp"
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
=======
#include "caffe/caffe.hpp"
#include "caffe/parallel.hpp"
#include "caffe/sgd_solvers.hpp"
>>>>>>> caffe-bvlc-dev/master

namespace caffe {

enum Op {
  copy,
  replace_cpu,
  replace_gpu,
  replace_cpu_diff,
  replace_gpu_diff
};

template<typename Dtype>
static void apply_buffers(const vector<Blob<Dtype>*>& blobs,
                          Dtype* buffer, size_t total_size, Op op) {
  Dtype* ptr = buffer;
  for (int i = 0; i < blobs.size(); ++i) {
    int size = blobs[i]->count();
    switch (op) {
      case copy: {
        // Init buffer to current values of blobs
        caffe_copy(size,
                   reinterpret_cast<const Dtype*>(blobs[i]->data()->cpu_data()),
                   ptr);
        break;
      }
      case replace_cpu:
        blobs[i]->data()->set_cpu_data(ptr);
        break;
      case replace_gpu:
        blobs[i]->data()->set_gpu_data(ptr);
        break;
      case replace_cpu_diff:
        blobs[i]->diff()->set_cpu_data(ptr);
        break;
      case replace_gpu_diff:
        blobs[i]->diff()->set_gpu_data(ptr);
        break;
    }
    ptr += size;
  }
  // total_size is at least one byte
  CHECK_EQ(total_size, (ptr == buffer ? 1 : ptr - buffer));
}

// Buffer size necessary to store given blobs
template<typename Dtype>
static size_t total_size(const vector<Blob<Dtype>*>& params) {
  size_t size = 0;
  for (int i = 0; i < params.size(); ++i)
    size += params[i]->count();
  // Size have at least one byte, otherwise cudaMalloc fails if net has no
  // learnable parameters.
  return (size > 0) ? size : 1;
}

template<typename Dtype>
Params<Dtype>::Params(shared_ptr<Solver<Dtype> > root_solver)
<<<<<<< HEAD
    : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
      data_(),
      diff_() {
=======
  : size_(total_size<Dtype>(root_solver->net()->learnable_params())),
    data_(),
    diff_() {
>>>>>>> caffe-bvlc-dev/master
}

template<typename Dtype>
GPUParams<Dtype>::GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device)
<<<<<<< HEAD
    : Params<Dtype>(root_solver) {
#ifndef CPU_ONLY
=======
  : Params<Dtype>(root_solver) {
>>>>>>> caffe-bvlc-dev/master
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));

  // Allocate device buffers
  CUDA_CHECK(cudaSetDevice(device));
  CUDA_CHECK(cudaMalloc(&data_, size_ * sizeof(Dtype)));

  // Copy blob values
  const vector<Blob<Dtype>*>& net =
<<<<<<< HEAD
      root_solver->net()->learnable_params();
=======
    root_solver->net()->learnable_params();
>>>>>>> caffe-bvlc-dev/master
  apply_buffers(net, data_, size_, copy);

  CUDA_CHECK(cudaMalloc(&diff_, size_ * sizeof(Dtype)));
  caffe_gpu_set(size_, Dtype(0), diff_);

  CUDA_CHECK(cudaSetDevice(initial_device));
<<<<<<< HEAD
#else
  NO_GPU;
#endif
=======
>>>>>>> caffe-bvlc-dev/master
}

template<typename Dtype>
GPUParams<Dtype>::~GPUParams() {
<<<<<<< HEAD
#ifndef CPU_ONLY
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
#endif
}

template<typename Dtype>
void GPUParams<Dtype>::configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
      solver->net()->learnable_params();
=======
  CUDA_CHECK(cudaFree(data_));
  CUDA_CHECK(cudaFree(diff_));
}

template<typename Dtype>
void GPUParams<Dtype>::Configure(Solver<Dtype>* solver) const {
  const vector<Blob<Dtype>*>& net =
    solver->net()->learnable_params();
>>>>>>> caffe-bvlc-dev/master
  apply_buffers(net, data_, size_, replace_gpu);
  apply_buffers(net, diff_, size_, replace_gpu_diff);
}

<<<<<<< HEAD
void DevicePair::compute(const vector<int> devices, vector<DevicePair>* pairs) {
#ifndef CPU_ONLY
  vector<int> remaining(devices);

  // Depth for reduction tree
  int remaining_depth = static_cast<int>(ceil(log2(remaining.size())));

  // Group GPUs by board
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        cudaDeviceProp a, b;
        CUDA_CHECK(cudaGetDeviceProperties(&a, remaining[i]));
        CUDA_CHECK(cudaGetDeviceProperties(&b, remaining[j]));
        if (a.isMultiGpuBoard && b.isMultiGpuBoard) {
          if (a.multiGpuBoardGroupID == b.multiGpuBoardGroupID) {
            pairs->push_back(DevicePair(remaining[i], remaining[j]));
            DLOG(INFO) << "GPU board: " << remaining[i] << ":" << remaining[j];
            remaining.erase(remaining.begin() + j);
            break;
          }
        }
      }
    }
  }
  ostringstream s;
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by boards, remaining: " << s.str();

  // Group by P2P accessibility
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      for (int j = i + 1; j < remaining.size(); ++j) {
        int access;
        CUDA_CHECK(
            cudaDeviceCanAccessPeer(&access, remaining[i], remaining[j]));
        if (access) {
          pairs->push_back(DevicePair(remaining[i], remaining[j]));
          DLOG(INFO) << "P2P pair: " << remaining[i] << ":" << remaining[j];
          remaining.erase(remaining.begin() + j);
          break;
        }
      }
    }
  }
  s.str("");
  for (int i = 0; i < remaining.size(); ++i) {
    s << (i ? ", " : "") << remaining[i];
  }
  DLOG(INFO) << "GPUs paired by P2P access, remaining: " << s.str();

  // Group remaining
  remaining_depth = ceil(log2(remaining.size()));
  for (int d = 0; d < remaining_depth; ++d) {
    for (int i = 0; i < remaining.size(); ++i) {
      pairs->push_back(DevicePair(remaining[i], remaining[i + 1]));
      DLOG(INFO) << "Remaining pair: " << remaining[i] << ":"
                 << remaining[i + 1];
      remaining.erase(remaining.begin() + i + 1);
    }
  }

  // Should only be the parent node remaining
  CHECK_EQ(remaining.size(), 1);

  pairs->insert(pairs->begin(), DevicePair(-1, remaining[0]));

  CHECK(pairs->size() == devices.size());
  for (int i = 0; i < pairs->size(); ++i) {
    CHECK((*pairs)[i].parent() != (*pairs)[i].device());
    for (int j = i + 1; j < pairs->size(); ++j) {
      CHECK((*pairs)[i].device() != (*pairs)[j].device());
    }
  }
#else
  NO_GPU;
#endif
}

//

template<typename Dtype>
P2PSync<Dtype>::P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                        P2PSync<Dtype>* parent, const SolverParameter& param)
    : GPUParams<Dtype>(root_solver, param.device_id()),
      parent_(parent),
      children_(),
      queue_(),
      initial_iter_(root_solver->iter()),
      solver_() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = param.device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent == NULL) {
    solver_ = root_solver;
  } else {
    Caffe::set_root_solver(false);
    solver_.reset(new WorkerSolver<Dtype>(param, root_solver.get()));
    Caffe::set_root_solver(true);
  }
  this->configure(solver_.get());
  solver_->add_callback(this);

  if (parent) {
    // Enable p2p access between devices
    const int peer = parent->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceEnablePeerAccess(peer, 0));
    } else {
      LOG(INFO)<< "GPU " << self << " does not have p2p access to GPU " << peer;
    }
    // Allocate receiving buffer on parent
    CUDA_CHECK(cudaSetDevice(peer));
    CUDA_CHECK(cudaMalloc(&parent_grads_, size_ * sizeof(Dtype)));
    CUDA_CHECK(cudaSetDevice(self));
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#else
  NO_GPU;
#endif
}

template<typename Dtype>
P2PSync<Dtype>::~P2PSync() {
#ifndef CPU_ONLY
  int initial_device;
  CUDA_CHECK(cudaGetDevice(&initial_device));
  const int self = solver_->param().device_id();
  CUDA_CHECK(cudaSetDevice(self));

  if (parent_) {
    CUDA_CHECK(cudaFree(parent_grads_));
    const int peer = parent_->solver_->param().device_id();
    int access;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&access, self, peer));
    if (access) {
      CUDA_CHECK(cudaDeviceDisablePeerAccess(peer));
    }
  }

  CUDA_CHECK(cudaSetDevice(initial_device));
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::InternalThreadEntry() {
  Caffe::SetDevice(solver_->param().device_id());
  CHECK(Caffe::root_solver());
  Caffe::set_root_solver(false);
  // See if there is a defined seed and reset random state if so
  if (solver_->param().random_seed() >= 0) {
    // Fetch random seed and modulate by device ID to make sure
    // everyone doesn't have the same seed.  We seem to have some
    // solver instability if we have everyone with the same seed
    Caffe::set_random_seed(
        solver_->param().random_seed() + solver_->param().device_id());
  }
  solver_->Step(solver_->param().max_iter() - initial_iter_);
}

template<typename Dtype>
void P2PSync<Dtype>::on_start() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#else
//  CHECK(false);
#endif

  // Wait for update from parent
  if (parent_) {
    P2PSync<Dtype> *parent = queue_.pop();
    CHECK(parent == parent_);
  }

  // Update children
  for (int i = children_.size() - 1; i >= 0; i--) {
    Dtype* src = data_;
    Dtype* dst = children_[i]->data_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == children_[i]->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    children_[i]->queue_.push(this);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::on_gradients_ready() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  CUDA_CHECK(cudaGetDevice(&device));
  CHECK(device == solver_->param().device_id());
#endif

  // Sum children gradients as they appear in the queue
  for (int i = 0; i < children_.size(); ++i) {
    P2PSync<Dtype> *child = queue_.pop();
    Dtype* src = child->parent_grads_;
    Dtype* dst = diff_;

#ifdef DEBUG
    bool ok = false;
    for (int j = 0; j < children_.size(); ++j) {
      if (child == children_[j]) {
        ok = true;
      }
    }
    CHECK(ok);
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == device);
#endif

    caffe_gpu_add(size_, src, dst, dst);
  }

  // Send gradients to parent
  if (parent_) {
    Dtype* src = diff_;
    Dtype* dst = parent_grads_;

#ifdef DEBUG
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, src));
    CHECK(attributes.device == device);
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, dst));
    CHECK(attributes.device == parent_->solver_->param().device_id());
#endif

    CUDA_CHECK(cudaMemcpyAsync(dst, src, size_ * sizeof(Dtype),  //
        cudaMemcpyDeviceToDevice, cudaStreamDefault));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    parent_->queue_.push(this);
  } else {
    // Loss functions divide gradients by the batch size, so to compensate
    // for split batch, the root solver divides by number of solvers.
    caffe_gpu_scal(size_, Dtype(1.0 / Caffe::solver_count()), diff_);
  }
#endif
}

template<typename Dtype>
void P2PSync<Dtype>::Prepare(const vector<int>& gpus,
            vector<shared_ptr<P2PSync<Dtype> > >* syncs) {
  // Pair devices for map-reduce synchronization
  vector<DevicePair> pairs;
  DevicePair::compute(gpus, &pairs);
  ostringstream s;
  for (int i = 1; i < pairs.size(); ++i) {
    s << (i == 1 ? "" : ", ") << pairs[i].parent() << ":" << pairs[i].device();
  }
  LOG(INFO)<< "GPUs pairs " << s.str();

  SolverParameter param(solver_->param());

  // Build the GPU tree by finding the parent for each solver
  for (int attempts = 0; attempts < pairs.size(); ++attempts) {
    for (int i = 1; i < pairs.size(); ++i) {
      if (!syncs->at(i).get()) {
        P2PSync<Dtype>* parent = NULL;
        for (int j = 0; j < syncs->size(); ++j) {
          P2PSync<Dtype>* sync = j == 0 ? this : syncs->at(j).get();
          if (sync) {
            const SolverParameter& p = sync->solver()->param();
            if (p.device_id() == pairs[i].parent()) {
              parent = sync;
            }
          }
        }
        if (parent) {
          param.set_device_id(pairs[i].device());
          syncs->at(i).reset(new P2PSync<Dtype>(solver_, parent, param));
          parent->children_.push_back((P2PSync<Dtype>*) syncs->at(i).get());
        }
      }
    }
  }
}

template<typename Dtype>
void P2PSync<Dtype>::Run(const vector<int>& gpus) {
  vector<shared_ptr<P2PSync<Dtype> > > syncs(gpus.size());
  Prepare(gpus, &syncs);

  LOG(INFO)<< "Starting Optimization";

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StartInternalThread();
  }

  // Run root solver on current thread
  solver_->Solve();

  for (int i = 1; i < syncs.size(); ++i) {
    syncs[i]->StopInternalThread();
=======
static int getDevice() {
  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  return device;
}

template<typename Dtype>
NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver)
  : GPUParams<Dtype>(solver, getDevice()),
    comm_(), solver_(solver), barrier_() {
  this->Configure(solver.get());
  Init();
}

template<typename Dtype>
NCCL<Dtype>::NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid)
  : GPUParams<Dtype>(solver, getDevice()),
    solver_(solver), barrier_() {
  this->Configure(solver.get());
  Caffe::set_multiprocess(true);
  ncclUniqueId nccl_uid;
  memcpy(&nccl_uid, &uid[0], NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
  NCCL_CHECK(ncclCommInitRank(&comm_,
                              Caffe::solver_count(),
                              nccl_uid,
                              Caffe::solver_rank()));
  Init();
}

template<typename Dtype>
void NCCL<Dtype>::Init() {
  if (solver_->param().layer_wise_reduce()) {
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
  }
}

template<typename Dtype>
NCCL<Dtype>::~NCCL() {
  if (solver_->param().layer_wise_reduce()) {
    CUDA_CHECK(cudaStreamDestroy(stream_));
  }
  if (comm_) {
    ncclCommDestroy(comm_);
  }
}

template<typename Dtype>
boost::barrier* NCCL<Dtype>::barrier() {
  return barrier_;
}
template<typename Dtype>
void NCCL<Dtype>::set_barrier(boost::barrier* value) {
  barrier_ = value;
}

template<typename Dtype>
void NCCL<Dtype>::InitSingleProcess(vector<NCCL<Dtype>*>* nccls) {
  ncclComm_t* comms = new ncclComm_t[nccls->size()];
  int* gpu_list = new int[nccls->size()];
  for (int i = 0; i < nccls->size(); ++i) {
    gpu_list[i] = (*nccls)[i]->solver_->param().device_id();
  }
  NCCL_CHECK(ncclCommInitAll(comms, static_cast<int>(nccls->size()), gpu_list));
  for (int i = 0; i < nccls->size(); ++i) {
    (*nccls)[i]->comm_ = comms[i];
  }
}

template<typename Dtype>
string NCCL<Dtype>::new_uid() {
  string uid;
  uid.resize(NCCL_UNIQUE_ID_BYTES);
  ncclUniqueId nccl_uid;
  NCCL_CHECK(ncclGetUniqueId(&nccl_uid));
  memcpy(&uid[0], &nccl_uid, NCCL_UNIQUE_ID_BYTES);  // NOLINT(caffe/alt_fn)
  return uid;
}

template<typename Dtype>
void NCCL<Dtype>::Broadcast() {
  if (barrier_) {  // NULL in multi process case
    barrier_->wait();
  }
  NCCL_CHECK(ncclBcast(data_, static_cast<int>(size_),
                       nccl::dataType<Dtype>::type, 0,
                       comm_, cudaStreamDefault));
  if (barrier_) {
    barrier_->wait();
  }
}

template<typename Dtype>
void NCCL<Dtype>::run(int layer) {
  CHECK(solver_->param().layer_wise_reduce());
  vector<shared_ptr<Blob<Dtype> > >& blobs =
    solver_->net()->layers()[layer]->blobs();
#ifdef DEBUG
  // Assert blobs are contiguous to reduce in one step (e.g. bias often small)
  for (int i = 1; i < blobs.size(); ++i) {
    CHECK_EQ(blobs[i - 1]->gpu_diff() + blobs[i - 1]->count(),
             blobs[i + 0]->gpu_diff());
  }
#endif
  if (blobs.size() > 0) {
    // Make sure default stream is done computing gradients. Could be
    // replaced by cudaEventRecord+cudaStreamWaitEvent to avoid
    // blocking the default stream, but it's actually slower.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));

    // Reduce asynchronously
    int size = 0;
    for (int i = 0; i < blobs.size(); ++i) {
      size += blobs[i]->count();
    }
    if (barrier_) {  // NULL in multi process case
      barrier_->wait();
    }
    NCCL_CHECK(ncclAllReduce(blobs[0]->mutable_gpu_diff(),
                             blobs[0]->mutable_gpu_diff(),
                             size,
                             nccl::dataType<Dtype>::type,
                             ncclSum, comm_, stream_));
    caffe_gpu_scal(size, (Dtype) 1.0 / Caffe::solver_count(),
                   blobs[0]->mutable_gpu_diff(), stream_);
  }
}

template<typename Dtype>
void NCCL<Dtype>::on_gradients_ready() {
  if (solver_->param().layer_wise_reduce()) {
    CHECK_EQ(solver_->net()->params().size(),
             solver_->net()->learnable_params().size())
      << "Layer-wise reduce is not supported for nets with shared weights.";

    // Make sure reduction is done before applying gradients
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  } else {
    if (barrier_) {  // NULL in multi process case
      barrier_->wait();
    }
    NCCL_CHECK(ncclAllReduce(diff_, diff_, static_cast<int>(size_),
                             nccl::dataType<Dtype>::type, ncclSum, comm_,
                             cudaStreamDefault));
    caffe_gpu_scal(static_cast<int>(size_),
                   (Dtype) 1.0 / Caffe::solver_count(), diff_);
  }
}

template<typename Dtype>
class Worker : public InternalThread {
 public:
  explicit Worker(shared_ptr<Solver<Dtype> > rank0, int device,
                  boost::barrier* barrier, vector<NCCL<Dtype>*>* nccls,
                  const char* restore)
    : rank0_(rank0), device_(device), barrier_(barrier),
      nccls_(nccls), restore_(restore) {
  }
  virtual ~Worker() {}

 protected:
  void InternalThreadEntry() {
    // Create solver and install callbacks
    SolverParameter param(rank0_->param());
    param.set_device_id(device_);
#ifdef DEBUG
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    CHECK_EQ(device, device_);
#endif
    param.set_type(rank0_->type());
    shared_ptr<Solver<Dtype> > s(SolverRegistry<Dtype>::CreateSolver(param));
    CHECK_EQ(s->type(), rank0_->type());
    if (restore_) {
      // Could not make NCCL broadcast solver state, it seems to crash
      // if called in a tight loop, regardless of barriers etc. so
      // restore all solvers from file.
      s->Restore(restore_);
    }
    NCCL<Dtype> nccl(s);
    nccl.set_barrier(barrier_);
    s->add_callback(&nccl);
    if (s->param().layer_wise_reduce()) {
      s->net()->add_after_backward(&nccl);
    }
    (*nccls_)[Caffe::solver_rank()] = &nccl;
    // Wait for other threads
    barrier_->wait();
    // Wait for NCCL init
    barrier_->wait();
    // Broadcast rank 0 state
    nccl.Broadcast();
    // Solve
    s->Step(param.max_iter() - s->iter());
    barrier_->wait();
#ifdef DEBUG
    // Check all solvers have same state
    SGDSolver<Dtype>* sa = static_cast<SGDSolver<Dtype>*>(rank0_.get());
    SGDSolver<Dtype>* sb = static_cast<SGDSolver<Dtype>*>(s.get());
    for (int h = 0; h < sa->history().size(); ++h) {
      CUDA_CHECK(cudaSetDevice(sa->param().device_id()));
      const Dtype* a = sa->history()[h]->cpu_data();
      CUDA_CHECK(cudaSetDevice(sb->param().device_id()));
      const Dtype* b = sb->history()[h]->cpu_data();
      for (int v = 0; v < sa->history()[h]->count(); ++v) {
        CHECK_DOUBLE_EQ(a[v], b[v]);
      }
    }
#endif
  }

  shared_ptr<Solver<Dtype> > rank0_;
  int device_;
  boost::barrier* barrier_;
  vector<NCCL<Dtype>*>* nccls_;
  const char* restore_;
};

template<typename Dtype>
void NCCL<Dtype>::Run(const vector<int>& gpus, const char* restore) {
  boost::barrier barrier(static_cast<int>(gpus.size()));
  vector<NCCL<Dtype>*> nccls(gpus.size());
  // Create workers
  vector<shared_ptr<Worker<Dtype> > > workers(gpus.size());
  for (int i = 1; i < gpus.size(); ++i) {
    CUDA_CHECK(cudaSetDevice(gpus[i]));
    Caffe::set_solver_rank(i);
    Worker<Dtype>* w = new Worker<Dtype>(solver_, gpus[i], &barrier,
                                         &nccls, restore);
    w->StartInternalThread();
    workers[i].reset(w);
  }
  CUDA_CHECK(cudaSetDevice(gpus[0]));
  Caffe::set_solver_rank(0);
  barrier_ = &barrier;
  solver_->add_callback(this);
  if (solver_->param().layer_wise_reduce()) {
    solver_->net()->add_after_backward(this);
  }
  nccls[0] = this;
  // Wait for workers
  barrier.wait();
  // Init NCCL
  InitSingleProcess(&nccls);
  barrier.wait();
  // Run first solver on current thread
  Broadcast();
  solver_->Solve();
  barrier.wait();  // Hangs without it when running tests
  // Wait for shutdown
  for (int i = 1; i < gpus.size(); ++i) {
    workers[i]->StopInternalThread();
>>>>>>> caffe-bvlc-dev/master
  }
}

INSTANTIATE_CLASS(Params);
INSTANTIATE_CLASS(GPUParams);
<<<<<<< HEAD
INSTANTIATE_CLASS(P2PSync);

}  // namespace caffe
=======
INSTANTIATE_CLASS(Worker);
INSTANTIATE_CLASS(NCCL);

}  // namespace caffe

#endif  // USE_NCCL
>>>>>>> caffe-bvlc-dev/master
