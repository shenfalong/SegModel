#ifndef CAFFE_SOLVER_HPP_
#define CAFFE_SOLVER_HPP_
#include <boost/function.hpp>
#include <string>
#include <vector>

#include "caffe/solver.hpp"
#include "caffe/net.hpp"


namespace caffe 
{


template <typename Dtype>
class Solver
{
 public:
 	Solver(){}
  static int iter() { return iter_; }
  static bool change_style() { return change_style_; }
  void share_weight(const shared_ptr<Net<Dtype> > net_, const shared_ptr<Net<Dtype> > test_net_);
 protected:
 
  static int iter_;
	static bool change_style_;
  DISABLE_COPY_AND_ASSIGN(Solver);
};


}  // namespace caffe

#endif  // CAFFE_SOLVER_HPP_
