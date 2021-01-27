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

RobustCost::RobustCost(RobustCostType costType) : mCostType(costType) {
  reset();
}

double RobustCost::weight(double rSq) {
  switch (mCostType) {
    case RobustCostType::L2: {
      return 1;
    }
    case RobustCostType::GNC_TLS: {
      // Implements eq. (14) of GNC paper
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
    case RobustCostType::L2: {
      break;
    }
    case RobustCostType::GNC_TLS: {
      mu = 0.01;
      break;
    }
    default: {
      throw std::runtime_error("calling reset with unknown cost type!");
    }
  }

  mIterationNumber = 0;
}

void RobustCost::updateGNCmu() {
  mIterationNumber++;
  if (mIterationNumber > mGNCMaxIters) {
    printf("Warning: GNC reached maximum iterations.");
    return;
  }

  switch (mCostType) {
    case RobustCostType::GNC_TLS: {
      mu = mGNCMuStep * mu;
      break;
    }
    default: {
      throw std::runtime_error("calling updateGNCmu with non-GNC cost function!");
    }
  }
}

}  // namespace DPGO