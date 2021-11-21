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

QuadraticProblem::QuadraticProblem(size_t nIn, size_t dIn, size_t rIn, const SparseMatrix &Q, const SparseMatrix &G)
    : n(nIn), d(dIn), r(rIn), mQ(Q), mG(G),
      M(new LiftedSEManifold(r, d, n)) {
  CHECK(r >= d);
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
  // Sanity check matrix dimensions
  CHECK_EQ(mG.rows(), (int) r);
  CHECK_EQ(mG.cols(), (int) ((d + 1) * n));
  CHECK_EQ(mQ.cols(), (int) ((d + 1) * n));
  CHECK_EQ(mQ.cols(), (int) ((d + 1) * n));
  constructPreconditioner();
}

QuadraticProblem::QuadraticProblem(const std::shared_ptr<PoseGraph> &pose_graph)
    : n(pose_graph->n()), d(pose_graph->d()), r(pose_graph->r()),
      pose_graph_(pose_graph),
      M(new LiftedSEManifold(pose_graph_->r(), pose_graph_->d(), pose_graph_->n())) {
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
  // Throw error if the pose graph cannot be initialized
  if (!pose_graph_->isInitialized()) {
    CHECK(pose_graph->initialize()) << "Input pose graph cannot be initialized!";
  }
  mQ = pose_graph_->quadraticMatrix();
  mG = pose_graph_->linearMatrix();
  constructPreconditioner();
}

QuadraticProblem::~QuadraticProblem() {
  delete M;
}

double QuadraticProblem::f(const Matrix &Y) const {
  CHECK_EQ((unsigned) Y.rows(), r);
  CHECK_EQ((unsigned) Y.cols(), (d + 1) * n);
  // returns 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace()
  return 0.5 * ((Y * mQ).cwiseProduct(Y)).sum() + (Y.cwiseProduct(mG)).sum();
}

double QuadraticProblem::f(ROPTLIB::Variable *x) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), r, (d + 1) * n);
  return 0.5 * ((X * mQ).cwiseProduct(X)).sum() + (X.cwiseProduct(mG)).sum();
}

void QuadraticProblem::EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const {
  Eigen::Map<const Matrix> X((double *) x->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> EG((double *) g->ObtainWriteEntireData(), r, (d + 1) * n);
  EG = X * mQ + mG;
}

void QuadraticProblem::EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                                     ROPTLIB::Vector *Hv) const {
  Eigen::Map<const Matrix> V((double *) v->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> HV((double *) Hv->ObtainWriteEntireData(), r, (d + 1) * n);
  HV = V * mQ;
}

void QuadraticProblem::PreConditioner(ROPTLIB::Variable *x,
                                      ROPTLIB::Vector *inVec,
                                      ROPTLIB::Vector *outVec) const {
  Eigen::Map<const Matrix> INVEC((double *) inVec->ObtainReadData(), r, (d + 1) * n);
  Eigen::Map<Matrix> OUTVEC((double *) outVec->ObtainWriteEntireData(), r, (d + 1) * n);
  OUTVEC = solver.solve(INVEC.transpose()).transpose();
  if (solver.info() == Eigen::Success) {
    M->getManifold()->Projection(x, outVec, outVec);  // Project output to the tangent space at x
  } else {
    printf("Preconditioner failed.\n");
    OUTVEC = INVEC;
  }
}

Matrix QuadraticProblem::RieGrad(const Matrix &Y) const {
  LiftedSEVariable Var(r, d, n);
  Var.setData(Y);
  LiftedSEVector EGrad(r, d, n);
  LiftedSEVector RGrad(r, d, n);
  EucGrad(Var.var(), EGrad.vec());
  M->getManifold()->Projection(Var.var(), EGrad.vec(), RGrad.vec());
  return RGrad.getData();
}

double QuadraticProblem::RieGradNorm(const Matrix &Y) const {
  return RieGrad(Y).norm();
}

void QuadraticProblem::constructPreconditioner() {
  // Update preconditioner
  SparseMatrix P = mQ;
  for (int i = 0; i < P.rows(); ++i) {
    P.coeffRef(i, i) += 1e-1;
  }
  solver.compute(P);
}

Matrix QuadraticProblem::readElement(const ROPTLIB::Element *element) const {
  return Eigen::Map<Matrix>((double *) element->ObtainReadData(), r, n * (d + 1));
}

void QuadraticProblem::setElement(ROPTLIB::Element *element, const Matrix *matrix) const {
  memcpy(element->ObtainWriteEntireData(), matrix->data(), sizeof(double) * r * (d + 1) * n);
}

}  // namespace DPGO
