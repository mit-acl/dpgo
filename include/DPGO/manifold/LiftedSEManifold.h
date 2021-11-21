/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/Poses.h>
#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

class LiftedSEManifold {
 public:
  /**
   * @brief Constructor
   * @param r
   * @param d
   * @param n
   */
  LiftedSEManifold(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Destructor
   */
  ~LiftedSEManifold();
  /**
   * @brief Get the underlying ROPTLIB product manifold
   * @return
   */
  ROPTLIB::ProductManifold *getManifold() { return MyManifold; }
  /**
   * @brief Utility function to project a given matrix onto this manifold
   * @param M
   * @return orthogonal projection of M onto this manifold
   */
  Matrix project(const Matrix &M) const;

 private:
  unsigned int r_, d_, n_;
  ROPTLIB::Stiefel *StiefelManifold;
  ROPTLIB::Euclidean *EuclideanManifold;
  ROPTLIB::ProductManifold *CartanManifold;
  ROPTLIB::ProductManifold *MyManifold;
};
}  // namespace DPGO

#endif