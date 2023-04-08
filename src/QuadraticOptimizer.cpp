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

QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem *p, ROptParameters params)
    : problem_(p),
      params_(params) {
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
  if (params_.method == ROptParameters::ROptMethod::RTR) {
    YOpt = trustRegion(Y);
  } else {
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
  if (gn0 < params_.gradnorm_tol) {
    return Yinit;
  }

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();
  ROPTLIB::RTRNewton Solver(problem_, VarInit.var());
  Solver.Stop_Criterion =
      ROPTLIB::StopCrit::GRAD_F;                                               // Stopping criterion based on absolute gradient norm
  Solver.Tolerance = params_.gradnorm_tol;                               // Tolerance associated with stopping criterion
  Solver.initial_Delta = params_.RTR_initial_radius;                         // Trust-region radius
  Solver.maximum_Delta = 5 * Solver.initial_Delta;                             // Maximum trust-region radius
  if (params_.verbose) {
    Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
  } else {
    Solver.Debug = ROPTLIB::DEBUGINFO::NOOUTPUT;
  }
  Solver.Max_Iteration = params_.RTR_iterations;
  Solver.Min_Inner_Iter = 0;
  Solver.Max_Inner_Iter = params_.RTR_tCG_iterations;
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
  if (params_.RGD_use_preconditioner) {
    problem_->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());
  }
  
  // Update
  M.getManifold()->ScaleTimesVector(VarInit.var(), -params_.RGD_stepsize, RGrad.vec(), RGrad.vec());
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
      (params_.verbose ? ROPTLIB::DEBUGINFO::DETAILED : ROPTLIB::DEBUGINFO::NOOUTPUT);
  Solver.Run();

  const auto *Yopt = dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());

  return VarOpt.getData();
}

}  // namespace DPGO
