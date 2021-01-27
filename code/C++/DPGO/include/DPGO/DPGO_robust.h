/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGOROBUST_H
#define DPGOROBUST_H

#include <iostream>
#include <cassert>
#include <DPGO/DPGO_utils.h>

namespace DPGO {

/**
 * @brief A list of supported robust cost functions
 */
enum RobustCostType {
  // L2 cost
  L2,

  // Graduated Non-Convexity (GNC) with truncated least squares (TLS)
  GNC_TLS,
};

/**
 * @brief Implementation of robust cost functions.
 *
 * Main references:
 * M-estimation:
 * Zhang, "Parameter Estimation Techniques: A Tutorial with Application to Conic Fitting"
 *
 * Graduated Non-Convexity (GNC):
 * Yang et al. "Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection"
 */
class RobustCost {
 public:
  RobustCost(RobustCostType costType);

  /**
   * @brief Compute measurement weight given current residual
   * @param rSq squared residual
   * @return weight
   */
  double weight(double rSq);

  /**
   * @brief Reset the mu parameter in GNC
   */
  void reset();

  /**
   * @brief Update the mu parameter in GNC
   */
  void updateGNCmu();

  /**
   * @brief Set maximum number of iterations for GNC
   * @param k
   */
  void setGNCMaxIteration(size_t k) { mGNCMaxIters = k; }

  /**
   * @brief Set GNC thresholds based on the quantile of chi-squared distribution
   * @param quantile
   * @param dimension
   */
  void setGNCThresholdAtQuantile(double quantile, size_t dimension) {
    assert(dimension == 2 || dimension == 3);
    assert(quantile > 0 && quantile < 1);
    mGNCBarcSq = chi2inv(quantile, dimension + 1);
    printf("Set GNC threshold at %f\n", mGNCBarcSq);
  }

  void setGNCThreshold(double sq) {
    assert(sq > 0);
    mGNCBarcSq = sq;
    printf("Set GNC threshold at %f\n", mGNCBarcSq);
  }

  /**
   * @brief Set GNC update factor
   * @param s
   */
  void setGNCMuStep(double s) { mGNCMuStep = s; }

 private:
  RobustCostType mCostType;

  // Parameters for graduated non-convexity (GNC)

  size_t mGNCMaxIters = 100;  // Maximum times to update mu

  double mGNCBarcSq = 1.0; // GNC thresholds

  double mGNCMuStep = 1.4; // Factor to update mu at each GNC iteration

  double mu = 0.05; // Mu parameter (only used in GNC)

  size_t mIterationNumber = 0; // Iteration number

};

}  // namespace DPGO

#endif