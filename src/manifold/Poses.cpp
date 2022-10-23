/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "DPGO/manifold/Poses.h"
#include "DPGO/DPGO_utils.h"
#include <glog/logging.h>

namespace DPGO {

LiftedPoseArray::LiftedPoseArray(unsigned int r, unsigned int d, unsigned int n) :
    r_(r), d_(d), n_(n) {
  X_ = Matrix::Zero(r_, (d_ + 1) * n_);
  Matrix Yinit = Matrix::Zero(r_, d_);
  Yinit.block(0, 0, d_, d_) = Matrix::Identity(d_, d_);
  for (unsigned int i = 0; i < n; ++i) {
    rotation(i) = Yinit;
    translation(i) = Vector::Zero(r_);
  }
}

Matrix LiftedPoseArray::getData() const {
  return X_;
}

void LiftedPoseArray::setData(const Matrix &X) {
  CHECK_EQ(X.rows(), r_);
  CHECK_EQ(X.cols(), (d_ + 1) * n_);
  X_ = X;
}

void LiftedPoseArray::checkData() const {
  for (unsigned i = 0; i < n_; ++i) {
    checkStiefelMatrix(rotation(i));
  }
}

Eigen::Ref<Matrix> LiftedPoseArray::pose(unsigned int index) {
  CHECK_LT(index, n_);
  return X_.block(0, index * (d_ + 1), r_, d_ + 1);
}

Matrix LiftedPoseArray::pose(unsigned int index) const {
  CHECK_LT(index, n_);
  return X_.block(0, index * (d_ + 1), r_, d_ + 1);
}

Eigen::Ref<Matrix> LiftedPoseArray::rotation(unsigned int index) {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.block(0, 0, r_, d_);
}

Matrix LiftedPoseArray::rotation(unsigned int index) const {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.block(0, 0, r_, d_);
}

Eigen::Ref<Vector> LiftedPoseArray::translation(unsigned int index) {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.col(d_);
}

Vector LiftedPoseArray::translation(unsigned int index) const {
  CHECK_LT(index, n_);
  auto Xi = X_.block(0, index * (d_ + 1), r_, d_ + 1);
  return Xi.col(d_);
}

double LiftedPoseArray::averageTranslationDistance(const LiftedPoseArray &poses1, const LiftedPoseArray &poses2) {
  CHECK_EQ(poses1.d(), poses2.d());
  CHECK_EQ(poses1.n(), poses2.n());
  double average_distance = 0;
  for (unsigned int i = 0; i < poses1.n(); ++i) {
    average_distance += (poses1.translation(i) - poses2.translation(i)).norm();
  }
  average_distance = average_distance / (double) poses1.n();
  return average_distance;
}

double LiftedPoseArray::maxTranslationDistance(const LiftedPoseArray &poses1, const LiftedPoseArray &poses2) {
  CHECK_EQ(poses1.d(), poses2.d());
  CHECK_EQ(poses1.n(), poses2.n());
  double max_distance = 0;
  for (unsigned int i = 0; i < poses1.n(); ++i) {
    max_distance = std::max(max_distance, (poses1.translation(i) - poses2.translation(i)).norm());
  }
  return max_distance;
}

Pose::Pose(const Matrix &T)
    : Pose(T.rows()) {
  CHECK_EQ(T.rows(), d_);
  CHECK_EQ(T.cols(), d_ + 1);
  setData(T);
}

Pose Pose::Identity(unsigned int d) {
  return Pose(d);
}

Pose Pose::identity() const {
  return Pose(d_);
}

Pose Pose::inverse() const {
  Matrix TInv = matrix().inverse();
  return Pose(TInv.block(0, 0, d_, d_ + 1));
}

Pose Pose::operator*(const Pose &other) const {
  CHECK_EQ(d(), other.d());
  Matrix Tr = matrix() * other.matrix();
  return Pose(Tr.block(0, 0, d_, d_ + 1));
}

Matrix Pose::matrix() const {
  Matrix T = Matrix::Identity(d_ + 1, d_ + 1);
  T.block(0, 0, d_, d_) = rotation();
  T.block(0, d_, d_, 1) = translation();
  return T;
}

}

