/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEVARIABLE_H
#define LIFTEDSEVARIABLE_H

#include <DPGO/DPGO_types.h>

#include <Eigen/Dense>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductElement.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

class LiftedSEVariable {
 public:
  LiftedSEVariable(int r, int d, int n);

  ~LiftedSEVariable();

  ROPTLIB::ProductElement* var() { return MyVariable; }

  /**
  Write data to matrix
  */
  Matrix getData();

  /**
        Set data from matrix
  */
  void setData(const Matrix& Y);

 private:
  ROPTLIB::StieVariable* StiefelVariable;
  ROPTLIB::EucVariable* EuclideanVariable;
  ROPTLIB::ProductElement* CartanVariable;
  ROPTLIB::ProductElement* MyVariable;
};
}  // namespace DPGO

#endif