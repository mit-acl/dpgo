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
  void setProblem(QuadraticProblem *p) { problem = p; }

  /**
  Turn on/off verbose output
  */
  void setVerbose(bool v) { verbose = v; }

  /**
  Set optimization algorithm
  */
  void setAlgorithm(ROPTALG alg) { algorithm = alg; }

  /**
  Set maximum step size
  */
  void setGradientDescentStepsize(double s) { gradientDescentStepsize = s; }

  /**
  Set number of trust region iterations
  */
  void setTrustRegionIterations(unsigned iter) { trustRegionIterations = iter; }

  /**
  Set tolerance of trust region
  */
  void setTrustRegionTolerance(double tol) { trustRegionTolerance = tol; }

  /**
   * @brief Set the initial trust region radius (default 1e1)
   * @param radius
   */
  void setTrustRegionInitialRadius(double radius) { trustRegionInitialRadius = radius; }

  /**
   * @brief Set the maximum number of inner tCG iterations
   * @param iter
   */
  void setTrustRegionMaxInnerIterations(int iter) { trustRegionMaxInnerIterations = iter; }

  /**
  Return optimization result
  */
  ROPTResult getOptResult() const { return result; };

 private:
  // Underlying Riemannian Optimization Problem
  QuadraticProblem *problem;

  // Optimization algorithm to be used
  ROPTALG algorithm;

  // Optimization result
  ROPTResult result;

  // step size (only for RGD)
  double gradientDescentStepsize;

  // Number of trust-region updates
  unsigned trustRegionIterations;

  // Tolerance for trust-region updates
  double trustRegionTolerance;

  // Initial trust region radius
  double trustRegionInitialRadius;

  // Maximum number of tCG iterations
  int trustRegionMaxInnerIterations;

  // Verbose flag
  bool verbose;

  // Apply RTR
  Matrix trustRegion(const Matrix &Yinit);

  // Apply a single RGD iteration with constant step size
  Matrix gradientDescent(const Matrix &Yinit);

  // Apply gradient descent with line search
  Matrix gradientDescentLS(const Matrix &Yinit);
};

}  // namespace DPGO

#endif
