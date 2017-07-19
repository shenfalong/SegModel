#ifndef CAFFE_SolverSecGAN_HPP_
#define CAFFE_SolverSecGAN_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/net.hpp"
#include "caffe/solver.hpp"

namespace caffe 
{

template <typename Dtype>
class SolverSecGAN : public Solver<Dtype>
{
 public:
 	explicit SolverSecGAN(const SolverParameter& param);
  void Solve(const char* all_state_file = NULL);
  void dispaly_loss(std::vector<Dtype> g_loss, std::vector<Dtype> d_loss);
  
 	void Snapshot();
  void Restore(const char* all_state_file);
	virtual ~SolverSecGAN() 
  {}
  
  inline const SolverParameter& param() const { return param_; }
  inline shared_ptr<Net<Dtype> > d_net() { return d_net_; }
  inline shared_ptr<Net<Dtype> > g_net() { return g_net_; }
  inline shared_ptr<Net<Dtype> > gd_net() { return g_net_; }
  inline shared_ptr<Net<Dtype> > d_interp_net() { return d_interp_net_; }
 protected:
  SolverParameter param_;
  shared_ptr<Net<Dtype> > dg_net_;
  shared_ptr<Net<Dtype> > d_net_;
  
  shared_ptr<Net<Dtype> > g_net_;
  shared_ptr<Net<Dtype> > gd_net_;
 
	shared_ptr<Net<Dtype> > d_interp_net_;


	int start_iter_;
	Dtype sum_loss_;


  DISABLE_COPY_AND_ASSIGN(SolverSecGAN);
};


}  // namespace caffe

#endif  // CAFFE_SolverSecGAN_HPP_
