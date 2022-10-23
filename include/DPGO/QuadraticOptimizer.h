/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef QUADRATICOPTIMIZER_H
#define QUADRATICOPTIMIZER_H

#include <DPGO/DPGO_types.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

namespace DPGO {

class QuadraticOptimizer {
 public:
  /**
   * @brief
   * @param p
   * @param params
   */
  QuadraticOptimizer(QuadraticProblem *p, ROptParameters params = ROptParameters());

  ~QuadraticOptimizer();

  /**
  Optimize from the given initial guess
  */
  Matrix optimize(const Matrix &Y);

  /**
   * Set optimization problem
   */
  void setProblem(QuadraticProblem *p) { problem_ = p; }

  /**
  Turn on/off verbose output
  */
  void setVerbose(bool v) { params_.verbose = v; }

  /**
  Set optimization algorithm
  */
  void setAlgorithm(ROptParameters::ROptMethod alg) { params_.method = alg; }

  /**
  Set maximum step size
  */
  void setRGDStepsize(double s) { params_.RGD_stepsize = s; }

  /**
  Set number of trust region iterations
  */
  void setRTRIterations(int iter) { params_.RTR_iterations = iter; }

  /**
  Set tolerance of trust region
  */
  void setGradientNormTolerance(double tol) { params_.gradnorm_tol = tol; }

  /**
   * @brief Set the initial trust region radius (default 1e1)
   * @param radius
   */
  void setRTRInitialRadius(double radius) { params_.RTR_initial_radius = radius; }

  /**
   * @brief Set the maximum number of inner tCG iterations
   * @param iter
   */
  void setRTRtCGIterations(int iter) { params_.RTR_tCG_iterations = iter; }

  /**
  Return optimization result
  */
  ROPTResult getOptResult() const { return result_; };

 private:
  // Underlying Riemannian Optimization Problem
  QuadraticProblem *problem_;

  // Optimization algorithm to be used
  ROptParameters params_;

  // Optimization result
  ROPTResult result_;

  // Timing
  SimpleTimer timer_;

  // Apply RTR
  Matrix trustRegion(const Matrix &Yinit);

  // Apply a single RGD iteration with constant step size
  Matrix gradientDescent(const Matrix &Yinit);

  // Apply gradient descent with line search
  Matrix gradientDescentLS(const Matrix &Yinit);
};

}  // namespace DPGO

#endif
