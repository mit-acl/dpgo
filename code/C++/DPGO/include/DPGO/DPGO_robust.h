/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGOROBUST_H
#define DPGOROBUST_H

#include <iostream>

namespace DPGO {

/**
 * @brief A base abstract class for M-estimators
 * Usage with different cost functions are implemented as derived classes
 *
 * Reference:
 * Zhang, "Parameter Estimation Techniques: A Tutorial with Application to Conic Fitting"
 */
class MEstimator {
 public:
  /**
  Cost function given the residual x
  */
  virtual double cost(double x) const = 0;

  /**
  Influence function (first derivative of cost)
  */
  virtual double influence(double x) const = 0;

  /**
  Weight function (influence over x)
  */
  virtual double weight(double x) const = 0;
};

class MEstimatorCauchy : public MEstimator {
 public:
  MEstimatorCauchy() : c(1.0) {}
  MEstimatorCauchy(double cIn) : c(cIn) {}

  double cost(double x) const override;
  double influence(double x) const override;
  double weight(double x) const override;

 private:
  double c;
};

class MEstimatorL2 : public MEstimator {
 public:
  double cost(double x) const override;
  double influence(double x) const override;
  double weight(double x) const override;
};

class MEstimatorTruncatedL2 : public MEstimator {
 public:
  MEstimatorTruncatedL2() : c(2.0) {}
  MEstimatorTruncatedL2(double cIn) : c(cIn) {}

  double cost(double x) const override;
  double influence(double x) const override;
  double weight(double x) const override;

 private:
  double c;
};

struct GNCParameters {
  // Maximum number of outer iterations
  size_t maxIters;

  // A factor is considered an inlier if factor.error() < barcSq.
  double barcSq;

  // Multiplicative factor to decrease/increase mu in GNC
  double muStep;

  // Default constructor
  GNCParameters(size_t maxIterations = 100, double barcSqIn = 1.0, double muStepIn = 1.4)
      : maxIters(maxIterations), barcSq(barcSqIn), muStep(muStepIn) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const GNCParameters &params) {
    os << "GNC parameters: " << std::endl;
    os << "Max iterations: " << params.maxIters << std::endl;
    os << "barcSq: " << params.barcSq << std::endl;
    os << "muStep: " << params.muStep << std::endl;
    return os;
  }
};

/**
 * @brief A base abstract class for Graduated Non-Convexity (GNC)
 * Usage with different cost functions (e.g., GM and TLS) are implemented as derived classes.
 *
 * Reference:
 * Yang et al. "Graduated Non-Convexity for Robust Spatial Perception: From Non-Minimal Solvers to Global Outlier Rejection"
 */
class GNC {
 public:
  /**
   * @brief Constructor of the abstract class. Note that derived class need to initialize Mu.
   */
  GNC(const GNCParameters &params) : mIterationNumber(0), mu(0.0), mParams(params) {}

  /**
   * @brief Compute measurement weight given input residual
   * Exact implementation depends on robust cost function
   * See Proposition 3 and 4 for GM and TLS costs, respectively.
   * @param r residual
   * @return weight
   */
  virtual double weight(double r) const = 0;

 protected:
  /**
   * @brief Update the Mu parameter in GNC.
   * Exact implementation depends on robust cost functions (see Remark 5 of GNC paper)
   */
  virtual void updateMu() = 0;

  // Current iteration
  size_t mIterationNumber;

  // Parameter controlling degree of non-convexity
  double mu;

  // GNC parameters
  GNCParameters mParams;
};

/**
 * @brief Graduated Non-Convexity using the truncated least square cost
 */
class GNC_TLS : public GNC {
 public:
  GNC_TLS(const GNCParameters &params);
  double weight(double r) const override;

 protected:
  void updateMu() override;
};

}  // namespace DPGO

#endif