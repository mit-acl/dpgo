/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/QuadraticProblem.h>

#include <iostream>

using namespace std;

/*Define the namespace*/
namespace DPGO {

QuadraticProblem::QuadraticProblem(size_t nIn, size_t dIn, size_t rIn):
    n(nIn), d(dIn), r(rIn),
    M(new LiftedSEManifold(r, d, n))
{
  assert(r >= d);
  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());
  setQ(SparseMatrix((d + 1) * n, (d + 1) * n));
  setG(SparseMatrix(r, (d + 1) * n));
}

QuadraticProblem::~QuadraticProblem() {
  delete M;
}

void QuadraticProblem::setQ(const SparseMatrix &QIn) {
  assert((unsigned) QIn.rows() == (d + 1) * n);
  assert((unsigned) QIn.cols() == (d + 1) * n);
  Q = QIn;

  // Update preconditioner
  SparseMatrix P = Q;
  for (int i = 0; i < P.rows(); ++i) {
    P.coeffRef(i, i) += 1.0;
  }
  solver.compute(P);
  if (solver.info() != Eigen::Success) {
    cout << "WARNING: preconditioner failed." << endl;
  }
}

void QuadraticProblem::setG(const SparseMatrix &GIn) {
  assert((unsigned) GIn.rows() == r);
  assert((unsigned) GIn.cols() == (d + 1) * n);
  G = GIn;
}

double QuadraticProblem::f(const Matrix &Y) const {
  assert((unsigned) Y.rows() == r);
  assert((unsigned) Y.cols() == (d + 1) * n);
  // returns 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace()
  return 0.5 * ((Y * Q).cwiseProduct(Y)).sum() + (Y.cwiseProduct(G)).sum();
}

double QuadraticProblem::f(ROPTLIB::Variable *x) const {
  return f(readElement(x));
}

void QuadraticProblem::EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const {
  Matrix EG = readElement(x) * Q + G;
  setElement(g, &EG);
}

void QuadraticProblem::EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                                     ROPTLIB::Vector *Hv) const {
  Matrix HVMat = readElement(v) * Q;
  setElement(Hv, &HVMat);
}

void QuadraticProblem::PreConditioner(ROPTLIB::Variable *x,
                                      ROPTLIB::Vector *inVec,
                                      ROPTLIB::Vector *outVec) const {
  Matrix HV = solver.solve(readElement(inVec).transpose()).transpose();
  if (solver.info() != Eigen::Success) {
    cout << "WARNING: Precon.solve() failed." << endl;
  }
  setElement(outVec, &HV);
  M->getManifold()->Projection(x, outVec, outVec);  // Project output to the tangent space at x
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

Matrix QuadraticProblem::readElement(const ROPTLIB::Element *element) const {
  return Eigen::Map<Matrix>((double *) element->ObtainReadData(), r, n * (d + 1));
}

void QuadraticProblem::setElement(ROPTLIB::Element *element, const Matrix *matrix) const {
  memcpy(element->ObtainWriteEntireData(), matrix->data(), sizeof(double) * r * (d + 1) * n);
}

}  // namespace DPGO
