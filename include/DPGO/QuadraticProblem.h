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
#include <DPGO/PoseGraph.h>

#include <Eigen/CholmodSupport>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <memory>

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
  /**
   * @brief Construct a quadratic optimization problem from a pose graph
   * @param pose_graph input pose graph must be initialized (or can be initialized) otherwise throw an runtime error
   */
  explicit QuadraticProblem(const std::shared_ptr<PoseGraph>& pose_graph);

  ~QuadraticProblem() override;

  /** Number of pose variables */
  unsigned int num_poses() const { return pose_graph_->n(); }

  /** Dimension (2 or 3) of estimation problem */
  unsigned int dimension() const { return pose_graph_->d(); }

  /** Relaxation rank in Riemannian optimization problem */
  unsigned int relaxation_rank() const { return pose_graph_->r(); }

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
  // The pose graph that represents the optimization problem
  std::shared_ptr<PoseGraph> pose_graph_;

  // Underlying manifold
  LiftedSEManifold *M;

  // Helper functions to convert between ROPTLIB::Element and Eigen Matrix
  Matrix readElement(const ROPTLIB::Element *element) const;
  void setElement(ROPTLIB::Element *element, const Matrix *matrix) const;
};

}  // namespace DPGO

#endif