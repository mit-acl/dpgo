#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>

#include <iostream>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, testStiefelGeneration) {
  Matrix Y = fixedStiefelVariable(3, 5);
  Matrix I = Matrix::Identity(3, 3);
  Matrix D = Y.transpose() * Y - I;
  ASSERT_LE(D.norm(), 1e-5);
}

TEST(testDPGO, testStiefelRepeat) {
  Matrix Y = fixedStiefelVariable(3, 5);
  for (size_t i = 0; i < 10; ++i) {
    Matrix Y_ = fixedStiefelVariable(3, 5);
    ASSERT_LE((Y_ - Y).norm(), 1e-5);
  }
}

TEST(testDPGO, testStiefelProjection) {
  size_t d = 3;
  size_t r = 5;
  Matrix I = Matrix::Identity(d, d);
  for (size_t j = 0; j < 50; ++ j) {
    Matrix M = Matrix::Random(r,d);
    Matrix Y = projectToStiefelManifold(M);
    Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}