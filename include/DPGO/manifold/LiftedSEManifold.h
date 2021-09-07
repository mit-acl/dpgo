/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include <DPGO/DPGO_types.h>
#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

class LiftedSEManifold {
 public:
  LiftedSEManifold(int r, int d, int n);

  ~LiftedSEManifold();

  ROPTLIB::ProductManifold* getManifold() { return MyManifold; }

  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  Matrix project(const Matrix& M) const;

 private:
  size_t r_;
  size_t d_;
  size_t n_;
  ROPTLIB::Stiefel* StiefelManifold;
  ROPTLIB::Euclidean* EuclideanManifold;
  ROPTLIB::ProductManifold* CartanManifold;
  ROPTLIB::ProductManifold* MyManifold;
};
}  // namespace DPGO

#endif