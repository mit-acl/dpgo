/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEVECTOR_H
#define LIFTEDSEVECTOR_H

#include <DPGO/DPGO_types.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

class LiftedSEVector {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
   */
  LiftedSEVector(int r, int d, int n);
  /**
   * @brief Destructor
   */
  ~LiftedSEVector();
  /**
   * @brief Get underlying ROPTLIB vector
   * @return
   */
  ROPTLIB::ProductElement* vec() { return MyVector; }
  /**
   * @brief Get data as Eigen matrix
   * @return
   */
  Matrix getData();
  /**
   * @brief Set data from Eigen matrix
   * @param Y
   */
  void setData(const Matrix& Y);

 private:
  ROPTLIB::StieVector* StiefelVector;
  ROPTLIB::EucVector* EuclideanVector;
  ROPTLIB::ProductElement* CartanVector;
  ROPTLIB::ProductElement* MyVector;
};
}  // namespace DPGO

#endif