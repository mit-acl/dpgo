#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/DPGO_robust.h>
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
  for (size_t j = 0; j < 50; ++j) {
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
  Matrix M = Matrix::Random(r, (d + 1) * n);
  Matrix X = Manifold.project(M);
  ASSERT_EQ(X.rows(), r);
  ASSERT_EQ(X.cols(), (d + 1) * n);
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
  for (int i = 0; i < numTrials; ++i) {
    double number = distribution(rng);
    if (number < threshold) count++;
  }
  double q = (double) count / numTrials;
  ASSERT_LE(abs(q - quantile), 0.01);
}

TEST(testDPGO, testRobustSingleRotationAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const double cbar = angular2ChordalSO3(0.5);  // approximately 30 deg
    std::vector<Matrix> RVec;
    RVec.push_back(RTrue);
    Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = Vector::Ones(1);
    robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa, cbar);
    checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDPGO, testRobustSingleRotationAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double tol = angular2ChordalSO3(0.02);
    const double cbar = angular2ChordalSO3(0.3);
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    std::vector<Matrix> RVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      if ((RRand - RTrue).norm() > 1.2 * cbar)  // Make sure that outlier is separated from the true rotation
        RVec.emplace_back(RRand);
    }
    Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = Vector::Ones(50);
    robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa, cbar);
    checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, tol);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}

TEST(testDPGO, testRobustSinglePoseAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<Matrix> RVec;
    RVec.push_back(RTrue);
    std::vector<Vector> tVec;
    tVec.push_back(tTrue);
    const auto kappa = 10000 * Vector::Ones(1);
    const auto tau = 100 * Vector::Ones(1);
    const double gnc_quantile = 0.9;
    const double gnc_barc = RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    Matrix ROpt;
    Vector tOpt;
    std::vector<size_t> inlierIndices;
    robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa, tau, gnc_barc);
    checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), 1e-8);
    ASSERT_LE((tOpt - tTrue).norm(), 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDPGO, testRobustSinglePoseAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double RMaxError = angular2ChordalSO3(0.02);
    const double tMaxError = 1e-2;
    const double gnc_quantile = 0.9;
    const double gnc_barc = RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    const double kappa = 10000;
    const double tau = 100;
    const auto kappa_vec = kappa * Vector::Ones(50);
    const auto tau_vec = tau * Vector::Ones(50);

    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<Matrix> RVec;
    std::vector<Vector> tVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
      tVec.emplace_back(tTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      Matrix tRand = Eigen::Vector3d::Random();
      double rSq = kappa * (RTrue - RRand).squaredNorm() + tau * (tTrue - tRand).squaredNorm();
      if (std::sqrt(rSq) > 1.2 * gnc_barc) { // Make sure that outliers are sufficiently far away from ground truth
        RVec.emplace_back(RRand);
        tVec.emplace_back(tRand);
      }
    }
    Matrix ROpt;
    Vector tOpt;
    std::vector<size_t> inlierIndices;
    robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa_vec, tau_vec, gnc_barc);
    checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), RMaxError);
    ASSERT_LE((tOpt - tTrue).norm(), tMaxError);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}