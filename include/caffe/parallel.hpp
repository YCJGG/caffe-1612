#ifndef CAFFE_PARALLEL_HPP_
#define CAFFE_PARALLEL_HPP_

<<<<<<< HEAD
#include <boost/date_time/posix_time/posix_time.hpp>

=======
#ifdef USE_NCCL

#include <boost/thread.hpp>

#include <string>
>>>>>>> caffe-bvlc-dev/master
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/solver.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/blocking_queue.hpp"
<<<<<<< HEAD
=======
#include "caffe/util/nccl.hpp"
>>>>>>> caffe-bvlc-dev/master

namespace caffe {

// Represents a net parameters. Once a net is created, its parameter buffers can
// be replaced by ones from Params, to allow parallelization. Params ensures
// parameters are allocated in one consecutive array.
template<typename Dtype>
class Params {
 public:
  explicit Params(shared_ptr<Solver<Dtype> > root_solver);
  virtual ~Params() {
  }

  inline size_t size() const {
    return size_;
  }
  inline Dtype* data() const {
    return data_;
  }
  inline Dtype* diff() const {
    return diff_;
  }

 protected:
  const size_t size_;           // Size of buffers
  Dtype* data_;                 // Network parameters
  Dtype* diff_;                 // Gradient

DISABLE_COPY_AND_ASSIGN(Params);
};

// Params stored in GPU memory.
template<typename Dtype>
class GPUParams : public Params<Dtype> {
 public:
  GPUParams(shared_ptr<Solver<Dtype> > root_solver, int device);
  virtual ~GPUParams();

<<<<<<< HEAD
  void configure(Solver<Dtype>* solver) const;
=======
  void Configure(Solver<Dtype>* solver) const;
>>>>>>> caffe-bvlc-dev/master

 protected:
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

<<<<<<< HEAD
class DevicePair {
 public:
  DevicePair(int parent, int device)
      : parent_(parent),
        device_(device) {
  }
  inline int parent() {
    return parent_;
  }
  inline int device() {
    return device_;
  }

  // Group GPUs in pairs, by proximity depending on machine's topology
  static void compute(const vector<int> devices, vector<DevicePair>* pairs);

 protected:
  int parent_;
  int device_;
};

// Synchronous data parallelism using map-reduce between local GPUs.
template<typename Dtype>
class P2PSync : public GPUParams<Dtype>, public Solver<Dtype>::Callback,
    public InternalThread {
 public:
  explicit P2PSync(shared_ptr<Solver<Dtype> > root_solver,
                   P2PSync<Dtype>* parent, const SolverParameter& param);
  virtual ~P2PSync();

  inline const shared_ptr<Solver<Dtype> >& solver() const {
    return solver_;
  }

  void Run(const vector<int>& gpus);
  void Prepare(const vector<int>& gpus,
               vector<shared_ptr<P2PSync<Dtype> > >* syncs);
  inline const int initial_iter() const { return initial_iter_; }

 protected:
  void on_start();
  void on_gradients_ready();

  void InternalThreadEntry();

  P2PSync<Dtype>* parent_;
  vector<P2PSync<Dtype>*> children_;
  BlockingQueue<P2PSync<Dtype>*> queue_;
  const int initial_iter_;
  Dtype* parent_grads_;
  shared_ptr<Solver<Dtype> > solver_;

=======
template<typename Dtype>
class NCCL : public GPUParams<Dtype>,
             public Solver<Dtype>::Callback,
             public Net<Dtype>::Callback {
 public:
  /**
   * Single process version.
   */
  explicit NCCL(shared_ptr<Solver<Dtype> > solver);
  /**
   * In multi-process settings, first create a NCCL id (new_uid), then
   * pass it to each process to create connected instances.
   */
  NCCL(shared_ptr<Solver<Dtype> > solver, const string& uid);
  ~NCCL();

  boost::barrier* barrier();
  void set_barrier(boost::barrier* value);

  /**
   * In single process settings, create instances without uids and
   * call this to connect them.
   */
  static void InitSingleProcess(vector<NCCL<Dtype>*>* nccls);

  static string new_uid();

  /**
   * Broadcast weights from rank 0 other solvers.
   */
  void Broadcast();

  /**
   * Single process multi-GPU.
   */
  void Run(const vector<int>& gpus, const char* restore);

 protected:
  void Init();
  void on_start() {}
  void run(int layer);  // Net callback
  void on_gradients_ready();

  ncclComm_t comm_;
  cudaStream_t stream_;

  shared_ptr<Solver<Dtype> > solver_;
  // Should not be necessary, https://github.com/NVIDIA/nccl/issues/37
  boost::barrier* barrier_;
>>>>>>> caffe-bvlc-dev/master
  using Params<Dtype>::size_;
  using Params<Dtype>::data_;
  using Params<Dtype>::diff_;
};

}  // namespace caffe

<<<<<<< HEAD
#endif
=======
#endif  // USE_NCCL
#endif  // header
>>>>>>> caffe-bvlc-dev/master
