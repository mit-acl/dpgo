/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticOptimizer.h>

#include <iostream>
#include <cassert>
#include <chrono>

#include "RSD.h"
#include "RTRNewton.h"
#include "SolversLS.h"

namespace DPGO {

QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem *p)
    : problem(p),
      algorithm(ROPTALG::RTR),
      gradientDescentStepsize(1e-3),
      trustRegionIterations(1),
      trustRegionTolerance(1e-2),
      trustRegionInitialRadius(1e1),
      trustRegionMaxInnerIterations(50),
      verbose(false) {
  result.success = false;
}

QuadraticOptimizer::~QuadraticOptimizer() = default;

Matrix QuadraticOptimizer::optimize(const Matrix &Y) {
  // Compute statistics before optimization
  result.fInit = problem->f(Y);
  result.gradNormInit = problem->RieGradNorm(Y);
  auto startTime = std::chrono::high_resolution_clock::now();

  // Optimize!
  Matrix YOpt;
  if (algorithm == ROPTALG::RTR) {
    YOpt = trustRegion(Y);
  } else {
    assert(algorithm == ROPTALG::RGD);
    YOpt = gradientDescent(Y);
  }

  // Compute statistics after optimization
  auto counter = std::chrono::high_resolution_clock::now() - startTime;
  result.elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();
  result.fOpt = problem->f(YOpt);
  result.gradNormOpt = problem->RieGradNorm(YOpt);
  result.relativeChange = sqrt((YOpt - Y).squaredNorm() / problem->num_poses());
  assert(result.fOpt <= result.fInit);

  return YOpt;
}

Matrix QuadraticOptimizer::trustRegion(const Matrix &Yinit) {
  unsigned r = problem->relaxation_rank();
  unsigned d = problem->dimension();
  unsigned n = problem->num_poses();

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();

  ROPTLIB::RTRNewton Solver(problem, VarInit.var());
  double initFunc = problem->f(VarInit.var());
  Solver.Stop_Criterion =
      ROPTLIB::StopCrit::GRAD_F;                                     // Stopping criterion based on absolute gradient norm
  Solver.Tolerance = trustRegionTolerance;                           // Tolerance associated with stopping criterion
  Solver.maximum_Delta = 10 * trustRegionInitialRadius;              // Maximum trust-region radius
  Solver.initial_Delta = trustRegionInitialRadius;                   // Initial trust-region radius
  if (verbose) {
    Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
  } else {
    Solver.Debug = ROPTLIB::DEBUGINFO::NOOUTPUT;
  }
  Solver.Max_Iteration = trustRegionIterations;
  Solver.Min_Inner_Iter = 0;
  Solver.Max_Inner_Iter = trustRegionMaxInnerIterations;
  Solver.theta = 1.0;  // Stopping condition of tCG (see 7.10 in Absil textbook)
  Solver.kappa = 0.1;  // Stopping condition of tCG (see 7.10 in Absil textbook)
  Solver.TimeBound = 5.0;
  Solver.Run();

  double funcDecrease = Solver.Getfinalfun() - initFunc;
  if (funcDecrease > -1e-8 && Solver.Getnormgf() > 1e-2) {
    // Optimization makes little progress while gradient norm is still large.
    // This means that the trust-region update is likely to be rejected. In this
    // case we need to increase number of max iterations and re-optimize.
    std::cout << "Trust-region update makes little progress. Running "
                 "more updates..."
              << std::endl;
    Solver.Max_Iteration = 10 * trustRegionIterations;
    Solver.Run();
  }

  // record tCG status
  result.tCGStatus = Solver.gettCGStatus();

  const auto *Yopt = dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());

  return VarOpt.getData();
}

Matrix QuadraticOptimizer::gradientDescent(const Matrix &Yinit) {
  unsigned r = problem->relaxation_rank();
  unsigned d = problem->dimension();
  unsigned n = problem->num_poses();

  LiftedSEManifold M(r, d, n);
  LiftedSEVariable VarInit(r, d, n);
  LiftedSEVariable VarNext(r, d, n);
  LiftedSEVector RGrad(r, d, n);
  VarInit.setData(Yinit);

  // Euclidean gradient
  problem->EucGrad(VarInit.var(), RGrad.vec());

  // Riemannian gradient
  M.getManifold()->Projection(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Preconditioning
  // problem->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());

  // Update
  M.getManifold()->ScaleTimesVector(VarInit.var(), -gradientDescentStepsize, RGrad.vec(), RGrad.vec());
  M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var(), 1.0);

  return VarNext.getData();
}

Matrix QuadraticOptimizer::gradientDescentLS(const Matrix &Yinit) {
  unsigned r = problem->relaxation_rank();
  unsigned d = problem->dimension();
  unsigned n = problem->num_poses();

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);

  ROPTLIB::RSD Solver(problem, VarInit.var());
  Solver.Stop_Criterion = ROPTLIB::StopCrit::GRAD_F;
  Solver.Tolerance = 1e-2;
  Solver.Max_Iteration = 10;
  Solver.Debug =
      (verbose ? ROPTLIB::DEBUGINFO::DETAILED : ROPTLIB::DEBUGINFO::NOOUTPUT);
  Solver.Run();

  const auto *Yopt = dynamic_cast<const ROPTLIB::ProductElement *>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());

  return VarOpt.getData();
}

}  // namespace DPGO
