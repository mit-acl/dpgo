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
Helper function to read a matrix from a .txt file
File format:
first row contains rows and cols
remaining rows store data
*/
Matrix read_matrix_from_file(const std::string &filename);

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
Given a vector of relative pose measurements, compute the chordal relaxation of
pose graph optimization
*/
Matrix chordalInitialization(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements);

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
Compute the whitened residual of a relative measurement
*/
double computeWhitenedResidual(const RelativeSEMeasurement &m,
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

}  // namespace DPGO

#endif