/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <glog/logging.h>

using namespace std;
using namespace ROPTLIB;

namespace DPGO {
LiftedSEManifold::LiftedSEManifold(unsigned int r, unsigned int d, unsigned int n) :
    r_(r), d_(d), n_(n) {
  StiefelManifold = new Stiefel((int) r, (int) d);
  StiefelManifold->ChooseStieParamsSet3();
  EuclideanManifold = new Euclidean((int) r);
  CartanManifold =
      new ProductManifold(2, StiefelManifold, 1, EuclideanManifold, 1);
  MyManifold = new ProductManifold(1, CartanManifold, n);
}

LiftedSEManifold::~LiftedSEManifold() {
  // Avoid memory leak
  delete StiefelManifold;
  delete EuclideanManifold;
  delete CartanManifold;
  delete MyManifold;
}

Matrix LiftedSEManifold::project(const Matrix &M) const {
  size_t expectedRows = r_;
  size_t expectedCols = (d_ + 1) * n_;
  CHECK_EQ(M.rows(), (int) expectedRows);
  CHECK_EQ(M.cols(), (int) expectedCols);
  Matrix X = M;
#pragma omp parallel for
  for (size_t i = 0; i < n_; ++i) {
    X.block(0, i * (d_ + 1), r_, d_) = projectToStiefelManifold(X.block(0, i * (d_ + 1), r_, d_));
  }
  return X;
}

}  // namespace DPGO
