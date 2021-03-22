/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef QUADRATICPROBLEM_H
#define QUADRATICPROBLEM_H

#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

#include "Problems/Problem.h"

/*Define the namespace*/
namespace DPGO {

/** This class implements a ROPTLIB problem with the following cost function:
    f(X) = 0.5*<Q, XtX> + <X,G>
    Q is the quadratic part with dimension (d+1)n-by-(d+1)n
    G is the linear part with dimension r-by-(d+1)n
*/
class QuadraticProblem : public ROPTLIB::Problem {
 public:
  QuadraticProblem(size_t nIn, size_t dIn, size_t rIn);

  ~QuadraticProblem() override;

  /** Number of pose variables */
  unsigned int num_poses() const { return n; }

  /** Dimension (2 or 3) of estimation problem */
  unsigned int dimension() const { return d; }

  /** Relaxation rank in Riemannian optimization problem */
  unsigned int relaxation_rank() const { return r; }

  /** get quadratic cost matrix */
  SparseMatrix getQ() const { return mQ; }

  /** get linear cost matrix */
  SparseMatrix getG() const { return mG; }

  /** set quadratic cost matrix */
  void setQ(const SparseMatrix &QIn);

  /** set linear cost matrix */
  void setG(const SparseMatrix &GIn);

  /**
   * @brief Evaluate objective function
   * @param Y
   * @return
   */
  double f(const Matrix &Y) const;

  /**
   * @brief Evaluate objective function
   * @param x
   * @return
   */
  double f(ROPTLIB::Variable *x) const override;

  /**
   * @brief Evaluate Euclidean gradient
   * @param x
   * @param g
   */
  void EucGrad(ROPTLIB::Variable *x, ROPTLIB::Vector *g) const override;

  /**
   * @brief Evaluate Hessian-vector product
   * @param x
   * @param v
   * @param Hv
   */
  void EucHessianEta(ROPTLIB::Variable *x, ROPTLIB::Vector *v,
                     ROPTLIB::Vector *Hv) const override;

  /**
   * @brief Evaluate preconditioner
   * @param x
   * @param inVec
   * @param outVec
   */
  void PreConditioner(ROPTLIB::Variable *x, ROPTLIB::Vector *inVec,
                      ROPTLIB::Vector *outVec) const override;

  /**
   * @brief Compute the Riemannian gradient at Y (represented in matrix form)
   * @param Y current point on the manifold (matrix form)
   * @return Riemannian gradient at Y as a matrix
   */
  Matrix RieGrad(const Matrix &Y) const;

  /**
   * @brief Compute Riemannian gradient norm at Y
   * @param Y current point on the manifold (matrix form)
   * @return Norm of the Riemannian gradient
   */
  double RieGradNorm(const Matrix &Y) const;

 private:
  // Number of poses
  const size_t n = 0;

  // Dimensionality of the Euclidean space
  const size_t d = 0;

  // The rank of the rank-restricted relaxation
  const size_t r = 0;

  /** The quadratic component of the cost function */
  SparseMatrix mQ;

  /** The linear component of the cost function */
  SparseMatrix mG;

  // ROPTLIB objects
  LiftedSEManifold *M;

  // Preconditioning solver
  Eigen::CholmodDecomposition<SparseMatrix> solver;

  // Helper functions to convert between ROPTLIB::Element and Eigen Matrix
  Matrix readElement(const ROPTLIB::Element *element) const;
  void setElement(ROPTLIB::Element *element, const Matrix *matrix) const;
};

}  // namespace DPGO

#endif