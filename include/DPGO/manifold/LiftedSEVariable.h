/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEVARIABLE_H
#define LIFTEDSEVARIABLE_H

#include <DPGO/DPGO_types.h>
#include <DPGO/manifold/Poses.h>

#include "Manifolds/Euclidean/Euclidean.h"
#include "Manifolds/ProductElement.h"
#include "Manifolds/Stiefel/Stiefel.h"

/*Define the namespace*/
namespace DPGO {

/**
 * @brief This object represent a collection of "lifted" poses X = [X1 X2 ... Xn]
 * that can be used by ROPTLIB to perform Riemannian optimization
 */
class LiftedSEVariable {
 public:
  /**
   * @brief Construct a default object
   * @param r relaxation rank
   * @param d dimension (2/3)
   * @param n number of poses
   */
  LiftedSEVariable(unsigned int r, unsigned int d, unsigned int n);
  /**
   * @brief Constructor from a lifted pose array object
   * @param poses
   */
  LiftedSEVariable(const LiftedPoseArray &poses);
  /**
   * @brief Copy constructor
   * @param other
   */
  LiftedSEVariable(const LiftedSEVariable &other);
  /**
   * @brief Destructor
   */
  ~LiftedSEVariable() = default;
  /**
   * @brief Copy assignment operator
   * @param other
   * @return
   */
  LiftedSEVariable &operator=(const LiftedSEVariable &other);
  /**
   * @brief Get relaxation rank
   * @return
   */
  unsigned int r() const { return r_; }
  /**
   * @brief Get dimension
   * @return
   */
  unsigned int d() const { return d_; }
  /**
   * @brief Get number of poses
   * @return
   */
  unsigned int n() const { return n_; }
  /**
   * @brief Obtain the variable as an ROPTLIB::ProductElement
   * @return
   */
  ROPTLIB::ProductElement *var() { return var_.get(); }
  /**
   * @brief Obtain the variable as an Eigen matrix
   * @return r by d+1 matrix [Y1 p1 ... Yn pn]
   */
  Matrix getData() const;
  /**
   * @brief Set this variable from an Eigen matrix
   * @param X r by d+1 matrix [Y1 p1 ... Yn pn]
   */
  void setData(const Matrix &X);
  /**
   * @brief Obtain the writable pose at the specified index, expressed as an r-by-(d+1) matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> pose(unsigned int index);
  /**
   * @brief Obtain the read-only pose at the specified index, expressed as an r-by-(d+1) matrix
   * @param index
   * @return
   */
  Matrix pose(unsigned int index) const;
  /**
   * @brief Obtain the writable rotation at the specified index, expressed as an r-by-d matrix
   * @param index
   * @return
   */
  Eigen::Ref<Matrix> rotation(unsigned int index);
  /**
   * @brief Obtain the read-only rotation at the specified index, expressed as an r-by-d matrix
   * @param index
   * @return
   */
  Matrix rotation(unsigned int index) const;
  /**
   * @brief Obtain the writable translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Eigen::Ref<Vector> translation(unsigned int index);
  /**
   * @brief Obtain the read-only translation at the specified index, expressed as an r dimensional vector
   * @param index
   * @return
   */
  Vector translation(unsigned int index) const;

 protected:
  // const dimensions
  unsigned int r_, d_, n_;
  // The actual content of this variable is stored inside a ROPTLIB::ProductElement
  std::unique_ptr<ROPTLIB::StieVariable> rotation_var_;
  std::unique_ptr<ROPTLIB::EucVariable> translation_var_;
  std::unique_ptr<ROPTLIB::ProductElement> pose_var_;
  std::unique_ptr<ROPTLIB::ProductElement> var_;
  // Internal view of the variable as an eigen matrix of dimension r-by-(d+1)*n
  Eigen::Map<Matrix> X_;
};

}  // namespace DPGO

#endif