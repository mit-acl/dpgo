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

QuadraticProblem::QuadraticProblem(unsigned int nIn,
                                   unsigned int dIn,
                                   unsigned int rIn,
                                   const SparseMatrix &QIn,
                                   const SparseMatrix &GIn)
    : Q(QIn), G(GIn), n(nIn), d(dIn), r(rIn) {

  M = new LiftedSEManifold(r, d, n);
  Variable = new LiftedSEVariable(r, d, n);
  Vector = new LiftedSEVector(r, d, n);
  HessianVectorProduct = new LiftedSEVector(r, d, n);

  ROPTLIB::Problem::SetUseGrad(true);
  ROPTLIB::Problem::SetUseHess(true);
  ROPTLIB::Problem::SetDomain(M->getManifold());

  SparseMatrix P = Q;
  P.diagonal().array() += 1.0;
  solver.compute(P);
  if (solver.info() != Eigen::Success) {
    cout << "WARNING: preconditioner failed." << endl;
  }
}

QuadraticProblem::~QuadraticProblem() {
  delete Variable;
  delete Vector;
  delete HessianVectorProduct;
  delete M;
}

double QuadraticProblem::f(const Matrix &Y) const {
  assert(Y.rows() == r);
  assert(Y.cols() == (d + 1) * n);
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
  Variable->setData(Y);
  LiftedSEVector EGrad(r, d, n);
  LiftedSEVector RGrad(r, d, n);
  EucGrad(Variable->var(), EGrad.vec());
  M->getManifold()->Projection(Variable->var(), EGrad.vec(), RGrad.vec());
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
