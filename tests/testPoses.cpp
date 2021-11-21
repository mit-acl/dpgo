//
// Created by yulun on 11/17/21.
//
#include <DPGO/manifold/Poses.h>
#include <DPGO/DPGO_utils.h>
#include <iostream>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, testLiftedPoseArray) {
  for (int trial = 0; trial < 50; ++trial) {
    int r = 5;
    int d = 3;
    int n = 3;
    LiftedPoseArray var(r, d, n);
    // Test setter and getter methods
    for (int i = 0; i < n; ++i) {
      auto Yi = randomStiefelVariable(d, r);
      auto pi = (i + 1) * Vector::Ones(r);
      var.rotation(i) = Yi;
      var.translation(i) = pi;
      ASSERT_LE((Yi - var.rotation(i)).norm(), 1e-6);
      ASSERT_LE((pi - var.translation(i)).norm(), 1e-6);
    }
    // Test copy constructor
    LiftedPoseArray var2(var);
    ASSERT_LE((var.getData() - var2.getData()).norm(), 1e-6);
  }
}

TEST(testDPGO, testLiftedPose) {
  int d = 3;
  int r = 5;
  for (int trial = 0; trial < 50; ++trial) {
    Matrix Xi = Matrix::Zero(r, d + 1);;
    Xi.block(0, 0, r, d) = randomStiefelVariable(d, r);
    Xi.col(d) = (trial + 1) * Vector::Ones(r);
    // Test constructor from Eigen matrix
    LiftedPose var(Xi);
    ASSERT_LE((Xi - var.getData()).norm(), 1e-6);
  }
}

TEST(testDPGO, testPoseIdentity) {
  int d = 3;
  Pose T(d);
  ASSERT_LE((T.identity().rotation() - Matrix::Identity(d, d)).norm(), 1e-6);
  ASSERT_LE((T.identity().translation() - Vector::Zero(d)).norm(), 1e-6);
  ASSERT_LE((T.identity().matrix() - Matrix::Identity(d + 1, d + 1)).norm(), 1e-6);
}

TEST(testDPGO, testPoseInverse) {
  for (int trial = 0; trial < 50; ++trial) {
    int d = 3;
    Pose T(d);
    T.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T.translation() = Vector::Random(d);
    auto TInv = T.inverse();
    ASSERT_LE((T.matrix() * TInv.matrix() - Matrix::Identity(d + 1, d + 1)).norm(), 1e-6);
    ASSERT_LE((TInv.matrix() * T.matrix() - Matrix::Identity(d + 1, d + 1)).norm(), 1e-6);
  }
}

TEST(testDPGO, testPoseMultiplication) {
  for (int trial = 0; trial < 50; ++trial) {
    int d = 3;
    Pose T1(d);
    T1.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T1.translation() = Vector::Random(d);
    Pose T2(d);
    T2.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    T2.translation() = Vector::Random(d);
    auto T = T1 * T2;
    ASSERT_LE((T1.matrix() * T2.matrix() - T.matrix()).norm(), 1e-6);
  }
}