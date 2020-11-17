/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticOptimizer.h>

#include <iostream>
#include <stdexcept>

#include "RSD.h"
#include "RTRNewton.h"
#include "SolversLS.h"

namespace DPGO {

QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem* p)
    : problem(p),
      algorithm(ROPTALG::RTR),
      gradientDescentStepsize(1e-3),
      trustRegionIterations(1),
      verbose(false) {}

QuadraticOptimizer::~QuadraticOptimizer() = default;

Matrix QuadraticOptimizer::optimize(const Matrix& Y) {
  if (algorithm == ROPTALG::RTR) {
    return trustRegion(Y);
  } else {
    assert(algorithm == ROPTALG::RGD);
    return gradientDescent(Y);
  }
}

Matrix QuadraticOptimizer::trustRegion(const Matrix& Yinit) {
  unsigned r = problem->relaxation_rank();
  unsigned d = problem->dimension();
  unsigned n = problem->num_poses();

  LiftedSEVariable VarInit(r, d, n);
  VarInit.setData(Yinit);
  VarInit.var()->NewMemoryOnWrite();

  ROPTLIB::RTRNewton Solver(problem, VarInit.var());
  double initFunc = problem->f(VarInit.var());
  Solver.Stop_Criterion =
      ROPTLIB::StopCrit::GRAD_F_0;  // Stoping criterion based on relative gradient norm
  Solver.Tolerance = 1e-6;     // Tolerance associated with stopping criterion
  Solver.maximum_Delta = 1e2;  // Maximum trust-region radius
  Solver.initial_Delta = 1e1;
  if (verbose) {
    Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
  } else {
    Solver.Debug = ROPTLIB::DEBUGINFO::NOOUTPUT;
  }
  Solver.Max_Iteration = trustRegionIterations;
  Solver.Min_Inner_Iter = 0;
  Solver.Max_Inner_Iter = 50;  
  Solver.TimeBound = 10;
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

  const auto* Yopt = dynamic_cast<const ROPTLIB::ProductElement*>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());

  return VarOpt.getData();
}

Matrix QuadraticOptimizer::gradientDescent(const Matrix& Yinit) {
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
  M.getManifold()->ScaleTimesVector(VarInit.var(), -gradientDescentStepsize,
                                    RGrad.vec(), RGrad.vec());
  M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var());

  return VarNext.getData();
}

Matrix QuadraticOptimizer::gradientDescentLS(const Matrix& Yinit) {
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

  const auto* Yopt = dynamic_cast<const ROPTLIB::ProductElement*>(Solver.GetXopt());
  LiftedSEVariable VarOpt(r, d, n);
  Yopt->CopyTo(VarOpt.var());
  if (verbose) {
    std::cout << "Initial objective value: " << problem->f(VarInit.var())
              << std::endl;
    std::cout << "Final objective value: " << Solver.Getfinalfun() << std::endl;
    std::cout << "Final gradient norm: " << Solver.Getnormgf() << std::endl;
  }

  return VarOpt.getData();
}

}  // namespace DPGO