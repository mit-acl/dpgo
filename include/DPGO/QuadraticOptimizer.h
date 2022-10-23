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
  QuadraticOptimizer(QuadraticProblem *p);

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
  void setVerbose(bool v) { verbose_ = v; }

  /**
  Set optimization algorithm
  */
  void setAlgorithm(ROptMethod alg) { algorithm_ = alg; }

  /**
  Set maximum step size
  */
  void setGradientDescentStepsize(double s) { gd_stepsize_ = s; }

  /**
  Set number of trust region iterations
  */
  void setTrustRegionIterations(unsigned iter) { trust_region_iterations_ = iter; }

  /**
  Set tolerance of trust region
  */
  void setTrustRegionTolerance(double tol) { trust_region_gradnorm_tol_ = tol; }

  /**
   * @brief Set the initial trust region radius (default 1e1)
   * @param radius
   */
  void setTrustRegionInitialRadius(double radius) { trust_region_initial_radius_ = radius; }

  /**
   * @brief Set the maximum number of inner tCG iterations
   * @param iter
   */
  void setTrustRegionMaxInnerIterations(int iter) { trust_region_max_inner_iterations_ = iter; }

  /**
  Return optimization result
  */
  ROPTResult getOptResult() const { return result_; };

 private:
  // Underlying Riemannian Optimization Problem
  QuadraticProblem *problem_;

  // Optimization algorithm to be used
  ROptMethod algorithm_;

  // Optimization result
  ROPTResult result_;

  // step size (only for RGD)
  double gd_stepsize_;

  // Number of trust-region updates
  unsigned trust_region_iterations_;

  // Tolerance for trust-region updates
  double trust_region_gradnorm_tol_;

  // Initial trust region radius
  double trust_region_initial_radius_;

  // Maximum number of tCG iterations
  int trust_region_max_inner_iterations_;

  // Verbose flag
  bool verbose_;

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
