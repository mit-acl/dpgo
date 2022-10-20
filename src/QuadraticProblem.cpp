/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticProblem.h>
#include <iostream>
#include <glog/logging.h>

using namespace std;

/*Define the namespace*/
namespace DPGO {

QuadraticProblem::QuadraticProblem(const std::shared_ptr<PoseGraph> &pose_graph)
    : pose_graph_(pose_graph),
      M(new LiftedSEManifold(pose_graph_->r(), pose_graph_->d(), pose_graph_->n())) {
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
}

QuadraticProblem::~QuadraticProblem() {
  delete M;
}

double QuadraticProblem::f(const Matrix &Y) const {
  CHECK_EQ((unsigned) Y.rows(), relaxation_rank());
  CHECK_EQ((unsigned) Y.cols(), (dimension() + 1) * num_poses());
  // returns 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace()
  return 0.5 * ((Y * pose_graph_->quadraticMatrix()).cwiseProduct(Y)).sum() +
      (Y.cwiseProduct(pose_graph_->linearMatrix())).sum();
}

double QuadraticProblem::f(ROPTLIB::Variable *x) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), relaxation_rank(), (dimension() + 1) * num_poses());
  return 0.5 * ((X * pose_graph_->quadraticMatrix()).cwiseProduct(X)).sum() +
      (X.cwiseProduct(pose_graph_->linearMatrix())).sum();
}

void QuadraticProblem::EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), relaxation_rank(), (dimension() + 1) * num_poses());
  Eigen::Map<Matrix> EG((double *) g->ObtainWriteEntireData(), relaxation_rank(), (dimension() + 1) * num_poses());
  EG = X * pose_graph_->quadraticMatrix() + pose_graph_->linearMatrix();
}

void QuadraticProblem::EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                                     ROPTLIB::Vector *Hv) const {
  Eigen::Map<const Matrix> V((double *) v->ObtainReadData(), relaxation_rank(), (dimension() + 1) * num_poses());
  Eigen::Map<Matrix> HV((double *) Hv->ObtainWriteEntireData(), relaxation_rank(), (dimension() + 1) * num_poses());
  HV = V * pose_graph_->quadraticMatrix();
}

void QuadraticProblem::PreConditioner(ROPTLIB::Variable *x,
                                      ROPTLIB::Vector *inVec,
                                      ROPTLIB::Vector *outVec) const {
  Eigen::Map<const Matrix>
      INVEC((double *) inVec->ObtainReadData(), relaxation_rank(), (dimension() + 1) * num_poses());
  Eigen::Map<Matrix>
      OUTVEC((double *) outVec->ObtainWriteEntireData(), relaxation_rank(), (dimension() + 1) * num_poses());
  if (pose_graph_->hasPreconditioner()) {
    OUTVEC = pose_graph_->preconditioner()->solve(INVEC.transpose()).transpose();
  } else {
    LOG(WARNING) << "Failed to compute preconditioner.";
  }
  M->getManifold()->Projection(x, outVec, outVec);  // Project output to the tangent space at x
}

Matrix QuadraticProblem::RieGrad(const Matrix &Y) const {
  LiftedSEVariable Var(relaxation_rank(), dimension(), num_poses());
  Var.setData(Y);
  LiftedSEVector EGrad(relaxation_rank(), dimension(), num_poses());
  LiftedSEVector RGrad(relaxation_rank(), dimension(), num_poses());
  EucGrad(Var.var(), EGrad.vec());
  M->getManifold()->Projection(Var.var(), EGrad.vec(), RGrad.vec());
  return RGrad.getData();
}

double QuadraticProblem::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

Matrix QuadraticProblem::readElement(const ROPTLIB::Element *element) const {
  return Eigen::Map<Matrix>((double *) element->ObtainReadData(),
                            relaxation_rank(),
                            num_poses() * (dimension() + 1));
}

void QuadraticProblem::setElement(ROPTLIB::Element *element, const Matrix *matrix) const {
  memcpy(element->ObtainWriteEntireData(),
         matrix->data(),
         sizeof(double) * relaxation_rank() * (dimension() + 1) * num_poses());
}

}  // namespace DPGO
