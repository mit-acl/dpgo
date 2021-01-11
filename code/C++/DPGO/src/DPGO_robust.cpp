/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_robust.h>

#include <cmath>

using namespace std;

namespace DPGO {

/*
 * L2
 */
double MEstimatorL2::cost(double x) const { return x * x / 2; }

double MEstimatorL2::influence(double x) const { return x; }

double MEstimatorL2::weight(double x) const { return 1.0; }

/*
 * Cauchy
 */
double MEstimatorCauchy::cost(double x) const {
  return (c * c / 2) * log(1 + (x / c) * (x / c));
}

double MEstimatorCauchy::influence(double x) const {
  return x / (1 + abs(x) / c);
}

double MEstimatorCauchy::weight(double x) const { return 1 / (1 + abs(x) / c); }

/*
 * Truncated L2
 */
double MEstimatorTruncatedL2::cost(double x) const {
  if (x > c) {
    return 0;
  }
  return x * x / 2;
}

double MEstimatorTruncatedL2::influence(double x) const {
  if (x > c) {
    return 0;
  }
  return x;
}

double MEstimatorTruncatedL2::weight(double x) const {
  if (x > c) {
    return 0;
  }
  return 1;
}

/*
 * GNC with truncated least squares cost
 */

GNC_TLS::GNC_TLS(const GNCParameters &params) : GNC(params) {
  // For TLS, initialize set Mu to be close to zero
  // TODO: proper distributed initialization of Mu
  mu = 0.1;
}

double GNC_TLS::weight(double r) const {
  // TODO: double check use of covariance
  // Implements eq. (14) of GNC paper
  double rSq = r * r;
  double upperBound = (mu + 1) / mu * mParams.barcSq;
  double lowerBound = mu / (mu + 1) * mParams.barcSq;
  if (rSq >= upperBound) {
    return 0;
  } else if (rSq <= lowerBound) {
    return 1;
  } else {
    return std::sqrt(mParams.barcSq * mu * (mu + 1) / rSq) - mu;
  }
}

void GNC_TLS::updateMu() {
  mIterationNumber++ ;
  mu = mParams.muStep * mu;
}
}  // namespace DPGO