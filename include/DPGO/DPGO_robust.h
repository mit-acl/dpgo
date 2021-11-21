/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGOROBUST_H
#define DPGOROBUST_H

#include <DPGO/DPGO_utils.h>
#include <iostream>
#include <glog/logging.h>

namespace DPGO {

/**
 * @brief Parameters for robust cost functions
 */
class RobustCostParameters {
 public:

  // Type of robust costs supported
  enum class Type {
    L2, // L2 (least squares)
    L1, // L1
    TLS, // truncated least squares
    Huber, // Huber loss
    GM, // Geman-McClure
    GNC_TLS, // Graduated Non-Convexity (GNC) with truncated least squares (TLS)
  };

  // Type of robust cost being used;
  Type costType;

  // GNC parameters
  unsigned GNCMaxNumIters;
  double GNCBarc;
  double GNCMuStep;
  double GNCInitMu;

  // Huber parameters
  double HuberThreshold;

  // Truncated least squares parameters
  double TLSThreshold;

  // Default constructor
  explicit RobustCostParameters(Type type = Type::L2,
                                unsigned gncMaxIters = 100,
                                double gncBarc = 10,
                                double gncMuStep = 1.4,
                                double gncInitMu = 1e-4,
                                double huberThresh = 3,
                                double TLSThresh = 10)
      : costType(type), GNCMaxNumIters(gncMaxIters), GNCBarc(gncBarc), GNCMuStep(gncMuStep), GNCInitMu(gncInitMu),
        HuberThreshold(huberThresh), TLSThreshold(TLSThresh) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const RobustCostParameters &params) {
    os << "Robust cost parameters: " << std::endl;
    os << "Cost function: " << robustCostName(params.costType) << std::endl;
    os << "GNC maximum iterations: " << params.GNCMaxNumIters << std::endl;
    os << "GNC mu step: " << params.GNCMuStep << std::endl;
    os << "GNC initial mu: " << params.GNCInitMu << std::endl;
    os << "GNC threshold (barc): " << params.GNCBarc << std::endl;
    os << "Huber threshold: " << params.HuberThreshold << std::endl;
    os << "TLS threshold: " << params.TLSThreshold << std::endl;
    return os;
  }
  /**
   * @brief String names for the supported robust cost functions
   * @param type
   * @return
   */
  static std::string robustCostName(RobustCostParameters::Type type);
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
  explicit RobustCost(const RobustCostParameters &params);

  /**
   * @brief Compute measurement weight given current residual
   * @param r residual (unsquared)
   * @return weight
   */
  double weight(double r) const;

  /**
   * @brief Reset the mu parameter in GNC
   */
  void reset();

  /**
   * @brief perform some auxiliary operations (e.g., update the mu parameter when GNC is used)
   */
  void update();

  /**
   * @brief Set error threshold based on the quantile of chi-squared distribution. This function only works for 3D measurements.
   * @param quantile
   * @param dimension
   * @return threshold
   */
  static double computeErrorThresholdAtQuantile(double quantile, size_t dimension) {
    CHECK_EQ((int) dimension, 3) << "quantile function currently only supports 3D problem.";
    CHECK_GT(quantile, 0);
    if (quantile < 1)
      return std::sqrt(chi2inv(quantile, 6));
    else
      return 1e5;
  }

 private:
  // Parameter settings
  const RobustCostParameters mParams;

  // GNC internal states
  size_t mGNCIteration = 0; // Iteration number
  double mu;                // Mu parameter

};

}  // namespace DPGO

#endif