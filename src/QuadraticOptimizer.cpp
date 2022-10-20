/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticOptimizer.h>
#include <glog/logging.h>
#include <iostream>

#include "RSD.h"
#include "RTRNewton.h"
#include "SolversLS.h"

namespace DPGO {

QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem *p)
    : problem_(p),
      algorithm_(ROPTALG::RTR),
      gd_stepsize_(1e-3),
      trust_region_iterations_(1),
      trust_region_gradnorm_tol_(1e-2),
      trust_region_initial_radius_(1e1),
      trust_region_max_inner_iterations_(50),
      verbose_(false) {
  result_.success = false;
}

QuadraticOptimizer::~QuadraticOptimizer() = default;

Matrix QuadraticOptimizer::optimize(const Matrix &Y) {
  // Compute statistics before optimization
  result_.fInit = problem_->f(Y);
  result_.gradNormInit = problem_->RieGradNorm(Y);
  timer_.tic();

  // Optimize!
  Matrix YOpt;
  if (algorithm_ == ROPTALG::RTR) {
    YOpt = trustRegion(Y);
  } else {
    CHECK_EQ(algorithm_, ROPTALG::RGD);
    YOpt = gradientDescent(Y);
  }

  // Compute statistics after optimization
  result_.elapsedMs = timer_.toc();
  result_.fOpt = problem_->f(YOpt);
  result_.gradNormOpt = problem_->RieGradNorm(YOpt);
  result_.success = true;
  // CHECK_LE(result_.fOpt, result_.fInit + 1e-5);

  return YOpt;
}

Matrix QuadraticOptimizer::trustRegion(const Matrix &Yinit) {
  unsigned r = problem_->relaxation_rank();
  unsigned d = problem_->dimension();
  unsigned n = problem_->num_poses();
  const double gn0 = problem_->RieGradNorm(Yinit);

  // No optimization if gradient norm already below threshold
  if (gn0 < trust_region_gradnorm_tol_) {
    return Yinit;
  }

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();
  ROPTLIB::RTRNewton Solver(problem_, VarInit.var());
  Solver.Stop_Criterion =
      ROPTLIB::StopCrit::GRAD_F;                                               // Stopping criterion based on absolute gradient norm
  Solver.Tolerance = trust_region_gradnorm_tol_;                               // Tolerance associated with stopping criterion
  Solver.initial_Delta = trust_region_initial_radius_;                         // Trust-region radius
  Solver.maximum_Delta = 5 * Solver.initial_Delta;                             // Maximum trust-region radius
  if (verbose_) {
    Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
  } else {
    Solver.Debug = ROPTLIB::DEBUGINFO::NOOUTPUT;
  }
  Solver.Max_Iteration = (int) trust_region_iterations_;
  Solver.Min_Inner_Iter = 0;
  Solver.Max_Inner_Iter = trust_region_max_inner_iterations_;
  Solver.TimeBound = 5.0;

  if (Solver.Max_Iteration == 1) {
    // Shrinking trust-region radius until step is accepted
    double radius = Solver.initial_Delta;
    int total_steps = 0;
    while (true) {
      Solver.initial_Delta = radius;
      Solver.maximum_Delta = radius;
      Solver.Run();
      if (Solver.latestStepAccepted()) {
        break;
      } else if (total_steps > 10) {
        printf("Too many RTR rejections. Returning initial guess.\n");
        return Yinit;
      } else {
        radius = radius / 4;
        total_steps++;
        printf("RTR step rejected. Shrinking trust-region radius to %f.\n", radius);
      }
    }
  } else {
    Solver.Run();
  }
  // record tCG status
  result_.tCGStatus = Solver.gettCGStatus();
  const auto *Yopt = dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());
  return VarOpt.getData();
}

Matrix QuadraticOptimizer::gradientDescent(const Matrix &Yinit) {
  unsigned r = problem_->relaxation_rank();
  unsigned d = problem_->dimension();
  unsigned n = problem_->num_poses();

  LiftedSEManifold M(r, d, n);
  LiftedSEVariable VarInit(r, d, n);
  LiftedSEVariable VarNext(r, d, n);
  LiftedSEVector RGrad(r, d, n);
  VarInit.setData(Yinit);

  // Euclidean gradient
  problem_->EucGrad(VarInit.var(), RGrad.vec());

  // Riemannian gradient
  M.getManifold()->Projection(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Preconditioning
  // problem->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Update
  M.getManifold()->ScaleTimesVector(VarInit.var(), -gd_stepsize_, RGrad.vec(), RGrad.vec());
  M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var());

  return VarNext.getData();
}

Matrix QuadraticOptimizer::gradientDescentLS(const Matrix &Yinit) {
  unsigned r = problem_->relaxation_rank();
  unsigned d = problem_->dimension();
  unsigned n = problem_->num_poses();

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);

  ROPTLIB::RSD Solver(problem_, VarInit.var());
  Solver.Stop_Criterion = ROPTLIB::StopCrit::GRAD_F;
  Solver.Tolerance = 1e-2;
  Solver.Max_Iteration = 10;
  Solver.Debug =
      (verbose_ ? ROPTLIB::DEBUGINFO::DETAILED : ROPTLIB::DEBUGINFO::NOOUTPUT);
  Solver.Run();

  const auto *Yopt = dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());

  return VarOpt.getData();
}

}  // namespace DPGO
