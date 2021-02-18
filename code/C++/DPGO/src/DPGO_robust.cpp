/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_robust.h>

#include <cmath>
#include <cassert>
#include <DPGO/DPGO_utils.h>

using namespace std;

namespace DPGO {

RobustCost::RobustCost(RobustCostType costType, const RobustCostParameters &params) :
    mCostType(costType), mParams(params) {
  reset();
}

double RobustCost::weight(double r) {
  switch (mCostType) {
    case RobustCostType::L2: {
      return 1;
    }
    case RobustCostType::L1: {
      return 1 / r;
    }
    case RobustCostType::Huber: {
      if (r < mParams.HuberThreshold) {
        return 1;
      } else {
        return mParams.HuberThreshold / r;
      }
    }
    case RobustCostType::TLS: {
      if (r < mParams.TLSThreshold) {
        return 1;
      } else {
        return 0;
      }
    }
    case RobustCostType::GM: {
      double a = 1 + r * r;
      return 1 / (a * a);
    }
    case RobustCostType::GNC_TLS: {
      // Implements eq. (14) of GNC paper
      double rSq = r * r;
      double mGNCBarcSq = mParams.GNCBarc * mParams.GNCBarc;
      double upperBound = (mu + 1) / mu * mGNCBarcSq;
      double lowerBound = mu / (mu + 1) * mGNCBarcSq;
      if (rSq >= upperBound) {
        return 0;
      } else if (rSq <= lowerBound) {
        return 1;
      } else {
        return std::sqrt(mGNCBarcSq * mu * (mu + 1) / rSq) - mu;
      }
    }
    default: {
      throw std::runtime_error("weight function for selected cost function is not implemented !");
    }
  }
}

void RobustCost::reset() {
  // Initialize the mu parameter in GNC, if used
  switch (mCostType) {
    case RobustCostType::GNC_TLS: {
      mu = mParams.GNCInitMu;
      mGNCIteration = 0;
      break;
    }
    default: {
      // do nothing
      break;
    }
  }

}

void RobustCost::update() {
  if (mCostType != RobustCostType::GNC_TLS) return;

  mGNCIteration++;
  if (mGNCIteration > mParams.GNCMaxNumIters) {
    printf("GNC: reached maximum iterations.");
    return;
  }

  switch (mCostType) {
    case RobustCostType::GNC_TLS: {
      mu = mParams.GNCMuStep * mu;
      break;
    }
    default: {
      throw std::runtime_error("Calling update for non-GNC cost function!");
    }
  }
}

}  // namespace DPGO