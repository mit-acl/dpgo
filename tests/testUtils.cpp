#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <iostream>
#include <random>

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
    Matrix M = Matrix::Random(r, d);
    Matrix Y = projectToStiefelManifold(M);
    Matrix D = Y.transpose() * Y - I;
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDPGO, testLiftedSEManifoldProjection) {
  int d = 3;
  int r = 5;
  int n = 100;
  LiftedSEManifold Manifold(r, d, n);
  Matrix M = Matrix::Random(r, (d+1)*n);
  Matrix X = Manifold.project(M);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d+1) * n);
  for (int i = 0; i < n; ++i) {
    Matrix Y = X.block(0, i * (d + 1), r, d);
    Matrix D = Y.transpose() * Y - Matrix::Identity(d, d);
    ASSERT_LE(D.norm(), 1e-5);
  }
}

TEST(testDPGO, testChi2Inv) {
  unsigned dof = 4;
  double quantile = 0.95;
  double threshold = chi2inv(quantile, dof);
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::chi_squared_distribution<double> distribution(dof);
  int numTrials = 100000;
  int count = 0;
  for (int i =0; i < numTrials; ++i) {
    double number = distribution(rng);
    if (number < threshold) count ++;
  }
  double q = (double) count / numTrials;
  ASSERT_LE(abs(q - quantile), 0.01);
}