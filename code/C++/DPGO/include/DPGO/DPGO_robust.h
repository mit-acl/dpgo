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
  L2, // L2 (least squares)
  L1, // L1
  TLS, // truncated least squares
  Huber, // Huber loss
  GM, // Geman-McClure
  GNC_TLS, // Graduated Non-Convexity (GNC) with truncated least squares (TLS)
};

const std::vector<std::string> RobustCostNames{"L2", "L1", "TLS", "Huber", "GM", "GNC_TLS"};

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
   * @param r residual (unsquared)
   * @return weight
   */
  double weight(double r);

  /**
   * @brief Reset the mu parameter in GNC
   */
  void reset();

  /**
   * @brief perform some auxiliary operations (e.g., update the mu parameter when GNC is used)
   */
  void update();

  /**
   * @brief Set maximum number of iterations for GNC
   * @param k
   */
  void setGNCMaxIteration(size_t k) { mGNCMaxIterations = k; }

  /**
   * @brief Set GNC thresholds based on the quantile of chi-squared distribution
   * @param quantile
   * @param dimension
   */
  void setGNCThresholdAtQuantile(double quantile, size_t dimension) {
    assert(dimension == 2 || dimension == 3);
    assert(quantile > 0 && quantile < 1);
    mGNCBarc = std::sqrt(chi2inv(quantile, dimension + 1));
    printf("Set GNC threshold at %f\n", mGNCBarc);
  }

  /**
   * @brief Set GNC threshold (unsquared) at given value
   * @param threshold
   */
  void setGNCThreshold(double threshold) {
    assert(threshold > 0);
    mGNCBarc = threshold;
    printf("Set GNC threshold at %f\n", mGNCBarc);
  }

  /**
   * @brief Set GNC update factor
   * @param s
   */
  void setGNCMuStep(double s) { mGNCMuStep = s; }

 private:
  RobustCostType mCostType;

  // #################################################
  // Parameters for Huber loss
  // #################################################
  double mHuberThreshold = 3;

  // #################################################
  // Parameters for TLS
  // #################################################
  double mTLSThreshold = 10;

  // #################################################
  // Parameters for graduated non-convexity (GNC)
  // #################################################
  size_t mGNCMaxIterations = 100;
  double mGNCBarc = 10; // GNC threshold
  double mGNCMuStep = 1.4; // Factor to update mu at each GNC iteration

  size_t mGNCIteration = 0; // Iteration number
  double mu = 0.05; // Mu parameter (only used in GNC)


};

}  // namespace DPGO

#endif