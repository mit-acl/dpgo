/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/manifold/LiftedSEVariable.h>

using namespace std;
using namespace ROPTLIB;

namespace DPGO {
LiftedSEVariable::LiftedSEVariable(int r, int d, int n) {
  StiefelVariable = new StieVariable(r, d);
  EuclideanVariable = new EucVariable(r);
  CartanVariable =
      new ProductElement(2, StiefelVariable, 1, EuclideanVariable, 1);
  MyVariable = new ProductElement(1, CartanVariable, n);
}

LiftedSEVariable::~LiftedSEVariable() {
  // Avoid memory leak
  delete StiefelVariable;
  delete EuclideanVariable;
  delete CartanVariable;
  delete MyVariable;
}

Matrix LiftedSEVariable::getData() {
  auto *T = dynamic_cast<ProductElement *>(MyVariable->GetElement(0));
  const int *sizes = T->GetElement(0)->Getsize();
  unsigned int r = sizes[0];
  unsigned int d = sizes[1];
  unsigned int n = MyVariable->GetNumofElement();
  return Eigen::Map<Matrix>((double *)MyVariable->ObtainReadData(), r,
                            n * (d + 1));
}

void LiftedSEVariable::setData(const Matrix &Y) {
  auto *T = dynamic_cast<ROPTLIB::ProductElement *>(MyVariable->GetElement(0));
  const int *sizes = T->GetElement(0)->Getsize();
  unsigned int r = sizes[0];
  unsigned int d = sizes[1];
  unsigned int n = MyVariable->GetNumofElement();
  assert(Y.rows() == r);
  assert(Y.cols() == (d+1) * n);

  // Copy array data from Eigen matrix to ROPTLIB variable
  const double *matrix_data = Y.data();
  double *prodvar_data = MyVariable->ObtainWriteEntireData();
  memcpy(prodvar_data, matrix_data, sizeof(double) * r * (d + 1) * n);
}

}  // namespace DPGO