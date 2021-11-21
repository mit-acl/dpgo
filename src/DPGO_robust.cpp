/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_robust.h>

#include <cmath>
#include <DPGO/DPGO_utils.h>

using namespace std;

namespace DPGO {

//"L2", "L1", "TLS", "Huber", "GM", "GNC_TLS"
std::string RobustCostParameters::robustCostName(RobustCostParameters::Type type) {
  std::string name;
  switch (type) {
    case Type::L2: {
      name = "L2";
      break;
    }
    case Type::L1: {
      name = "L1";
      break;
    }
    case Type::TLS: {
      name = "TLS";
      break;
    }
    case Type::Huber: {
      name = "Huber";
      break;
    }
    case Type::GM: {
      name = "GM";
      break;
    }
    case Type::GNC_TLS: {
      name = "GNC_TLS";
      break;
    }
  }
  return name;
}

RobustCost::RobustCost(const RobustCostParameters &params) :
    mParams(params), mu(params.GNCInitMu) {
  reset();
}

double RobustCost::weight(double r) const {
  switch (mParams.costType) {
    case RobustCostParameters::Type::L2: {
      return 1;
    }
    case RobustCostParameters::Type::L1: {
      return 1 / r;
    }
    case RobustCostParameters::Type::Huber: {
      if (r < mParams.HuberThreshold) {
        return 1;
      } else {
        return mParams.HuberThreshold / r;
      }
    }
    case RobustCostParameters::Type::TLS: {
      if (r < mParams.TLSThreshold) {
        return 1;
      } else {
        return 0;
      }
    }
    case RobustCostParameters::Type::GM: {
      double a = 1 + r * r;
      return 1 / (a * a);
    }
    case RobustCostParameters::Type::GNC_TLS: {
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
  switch (mParams.costType) {
    case RobustCostParameters::Type::GNC_TLS: {
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
  if (mParams.costType != RobustCostParameters::Type::GNC_TLS) return;

  mGNCIteration++;
  if (mGNCIteration > mParams.GNCMaxNumIters) {
    printf("GNC: reached maximum iterations.");
    return;
  }

  switch (mParams.costType) {
    case RobustCostParameters::Type::GNC_TLS: {
      mu = mParams.GNCMuStep * mu;
      break;
    }
    default: {
      throw std::runtime_error("Calling update for non-GNC cost function!");
    }
  }
}

}  // namespace DPGO