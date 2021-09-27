/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGOUTILS_H
#define DPGOUTILS_H

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativeSEMeasurement.h>

#include <Eigen/Dense>
#include <Eigen/SVD>

// ROPTLIB includes
#include "Manifolds/Stiefel/Stiefel.h"

namespace DPGO {
/**
 * @brief Write a dense Eigen matrix to file
 * @param M
 * @param filename
 */
void writeMatrixToFile(const Matrix &M, const std::string &filename);

/**
 * @brief Write a sparse matrix to file
 * @param M
 * @param filename
 */
void writeSparseMatrixToFile(const SparseMatrix &M, const std::string &filename);

/**
Helper function to read a dataset in .g2o format
*/
std::vector<RelativeSEMeasurement> read_g2o_file(const std::string &filename,
                                                 size_t &num_poses);

/**
Helper function to construct connection laplacian matrix in SE(d)
*/
void constructOrientedConnectionIncidenceMatrixSE(
    const std::vector<RelativeSEMeasurement> &measurements, SparseMatrix &AT,
    DiagonalMatrix &OmegaT);

/**
Helper function to construct connection laplacian matrix in SE(d)
*/
SparseMatrix constructConnectionLaplacianSE(
    const std::vector<RelativeSEMeasurement> &measurements);

/**
Given a vector of relative pose measurements, this function computes and returns
the B matrices defined in equation (69) of the tech report
*/
void constructBMatrices(const std::vector<RelativeSEMeasurement> &measurements,
                        SparseMatrix &B1, SparseMatrix &B2, SparseMatrix &B3);

/**
 * @brief Initialize local trajectory estimate from chordal relaxation
 * @param dimension
 * @param num_poses
 * @param measurements
 * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
 */
Matrix chordalInitialization(size_t dimension,
                             size_t num_poses,
                             const std::vector<RelativeSEMeasurement> &measurements);

/**
 * @brief Initialize local trajectory estimate from odometry
 * @param dimension
 * @param num_poses
 * @param odometry A vector of odometry measurement
 * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
 */
Matrix odometryInitialization(size_t dimension, size_t num_poses, const std::vector<RelativeSEMeasurement> &odometry);

/**
Given the measurement matrices B1 and B2 and a matrix R of rotational state
estimates, this function computes and returns the corresponding optimal
translation estimates
*/
Matrix recoverTranslations(const SparseMatrix &B1, const SparseMatrix &B2,
                           const Matrix &R);

/**
Project a given matrix to the rotation group
*/
Matrix projectToRotationGroup(const Matrix &M);

/**
 * @brief project an input matrix M to the Stiefel manifold
 * @param M
 * @return orthogonal projection of M to Stiefel manifold
 */
Matrix projectToStiefelManifold(const Matrix &M);

/**
Generate a random element of the Stiefel element
The returned value is guaranteed to be the same for each d and r
*/
Matrix fixedStiefelVariable(unsigned d, unsigned r);

/**
 * @brief Compute the error term (weighted squared residual)
 * @param m measurement
 * @param R1 rotation of first pose
 * @param t1 translation of first pose
 * @param R2 rotation of second pose
 * @param t2 translation of second pose
 * @return
 */
double computeMeasurementError(const RelativeSEMeasurement &m,
                               const Matrix &R1, const Matrix &t1,
                               const Matrix &R2, const Matrix &t2);

/**
 * @brief Quantile of chi-squared distribution with given degrees of freedom at probability alpha.
 * Equivalent to chi2inv in Matlab.
 * @param quantile
 * @param dof
 * @return
 */
double chi2inv(double quantile, size_t dof);

/**
 * @brief For SO(3), convert angular distance in radian to chordal distance
 * @param rad input angular distance in radian
 * @return
 */
double angular2ChordalSO3(double rad);

/**
 * @brief Verify that the input matrix is a valid rotation
 * @param R
 */
void checkRotationMatrix(const Matrix &R);

/**
 * @brief Single translation averaging using the Euclidean distance
 * @param tOpt
 * @param tVec
 * @param tau
 */
void singleTranslationAveraging(Vector &tOpt,
                                const std::vector<Vector> &tVec,
                                const Vector &tau = Vector::Ones(0));

/**
 * @brief Single rotation averaging with the chordal distance
 * @param ROpt
 * @param RVec
 * @param kappa
 */
void singleRotationAveraging(Matrix &ROpt,
                             const std::vector<Matrix> &RVec,
                             const Vector &kappa = Vector::Ones(0));

/**
 * @brief Single pose averaging with chordal distance
 * @param ROpt
 * @param tOpt
 * @param RVec
 * @param tVec
 * @param kappa
 * @param tau
 */
void singlePoseAveraging(Matrix &ROpt, Vector &tOpt,
                         const std::vector<Matrix> &RVec,
                         const std::vector<Vector> &tVec,
                         const Vector &kappa = Vector::Ones(0),
                         const Vector &tau = Vector::Ones(0));

/**
 * @brief Robust single rotation averaging using GNC
 * @param ROpt output rotation matrix
 * @param inlierIndices output inlier indices
 * @param RVec input rotation matrices
 * @param kappaVec weights associated with rotation matrices
 * @param errorThreshold max error threshold under Langevin noise distribution
 */
void robustSingleRotationAveraging(Matrix &ROpt,
                                   std::vector<size_t> &inlierIndices,
                                   const std::vector<Matrix> &RVec,
                                   const Vector &kappa = Vector::Ones(0),
                                   double errorThreshold = 0.1);

/**
 * @brief Robust single pose averaging using GNC
 * @param ROpt
 * @param tOpt
 * @param inlierIndices
 * @param RVec
 * @param tVec
 * @param kappa
 * @param tau
 * @param errorThreshold max error threshold under Langevin noise distribution
 */
void robustSinglePoseAveraging(Matrix &ROpt, Vector &tOpt,
                               std::vector<size_t> &inlierIndices,
                               const std::vector<Matrix> &RVec,
                               const std::vector<Vector> &tVec,
                               const Vector &kappa = Vector::Ones(0),
                               const Vector &tau = Vector::Ones(0),
                               double errorThreshold = 0.1);

}  // namespace DPGO

#endif