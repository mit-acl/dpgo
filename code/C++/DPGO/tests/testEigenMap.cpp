#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, EigenMap) {
  size_t d = 3;
  size_t n = 10;
  LiftedSEVariable x(d, d, n);
  x.var()->RandInManifold();

  // View the internal memory of x as a read-only eigen matrix
  Eigen::Map<const Matrix> xMatConst((double *) x.var()->ObtainReadData(), d, (d + 1) * n);
  ASSERT_LE((xMatConst - x.getData()).norm(), 1e-4);

  // View the internal memory of x as a writable eigen matrix
  Eigen::Map<Matrix> xMat((double *) x.var()->ObtainWriteEntireData(), d, (d + 1) * n);

  // Modify x through eigen map
  for (size_t i = 0; i < n; ++i) {
    xMat.block(0, i * (d+1),     d, d) = Matrix::Identity(d, d);
    xMat.block(0, i * (d+1) + d, d, 1) = Matrix::Zero(d, 1);
  }

  // Check that the internal value of x is modified accordingly
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);

  xMat = Matrix::Random(d, (d + 1) *n);
  ASSERT_LE((xMat - x.getData()).norm(), 1e-4);
}
