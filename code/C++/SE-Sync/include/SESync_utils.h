#ifndef _SESYNC_UTILS_H_
#define _SESYNC_UTILS_H_

#include "SESync_types.h"
#include "ProductElement.h"

#include <string>
#include <vector>


namespace SESync {

// Forward declaration of necessary types here
class RelativePoseMeasurement;
class CartanSyncVariable;

typedef std::vector<RelativePoseMeasurement> measurements_t;

measurements_t read_g2o_file(const std::string& filename, size_t& num_poses);

/** Given a vector of relative pose measurements, this function computes and
 * returns the corresponding SE(d) connection incidence matrix */
void construct_oriented_connection_incidence_matrix_T(
    const std::vector<RelativePoseMeasurement>& measurements,
    SparseMatrix& AT, DiagonalMatrix& OmegaT );
//construct_oriented_connection_incidence_matrix_T(const measurements_t& measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the corresponding SE(d) connection precision matrix */
//SparseMatrix
//construct_connection_precision_matrix_T(const measurements_t& measurements);

/** Given the SE(d) connection incidence and precision matrices, this function
 * computes and returns the corresponding SE(d) connection Laplacian */
SparseMatrix
construct_connection_Laplacian_T(
    const SparseMatrix& AT, const DiagonalMatrix& OmegaT );

// TODO: Comment this
SparseMatrix
construct_connection_Laplacian_T( const measurements_t& measurements );

/** Given a vector of relative pose measurements, this function computes and
 * returns the corresponding rotational connection Laplacian */
SparseMatrix
construct_rotational_connection_Laplacian(const measurements_t& measurements);

/** Given the SE(d) connection incidence and precision matrices, this function
 * computes and returns the corresponding SO(3) connection Laplacian */
SparseMatrix
construct_rotational_connection_Laplacian(
    const SparseMatrix& AT, const DiagonalMatrix& OmegaT,
    const size_t d );

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated oriented incidence matrix A */
SparseMatrix
construct_oriented_incidence_matrix(const measurements_t& measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated diagonal matrix of translational measurement
 * precisions */
DiagonalMatrix
construct_translational_precision_matrix(const measurements_t& measurements);

/** Given a vector of relative pose measurements, this function computes and
 * returns the associated matrix of raw translational measurements */
SparseMatrix
construct_translational_data_matrix(const measurements_t& measurements);

/** Given a vector of relative pose measurements, this function computes and returns the B matrices defined in equation (69) of the tech report */
void construct_B_matrices(const std::vector<RelativePoseMeasurement>& measurements, SparseMatrix& B1, SparseMatrix& B2, SparseMatrix& B3);

/** Given the rotational connection Laplacian encoding the set of rotational
 * observations for a pose-graph SLAM problem on SE(d), this function computes
 * and returns the corresponding chordal initialization for the rotational
 * states */
Matrix chordal_initialization(
    const SparseMatrix& rotational_connection_Laplacian,
    unsigned int d, unsigned int max_iterations = 10000,
    double precision = 1e-6);

/** Given the measurement matrix B3 defined in equation (69c) of the tech report and the problem dimension d, this function computes and returns the corresponding chordal initialization for the rotational states */
Matrix chordal_initialization(unsigned int d, const SparseMatrix& B3);

/** Given the measurement matrices B1 and B2 and a matrix R of rotational state estimates, this function computes and returns the corresponding optimal translation estimates */
Matrix recover_translations(const SparseMatrix& B1, const SparseMatrix& B2, const Matrix& R);

/** Given a square d x d matrix, this function returns a closest element of
 * SO(d) */
Matrix project_to_SOd(const Matrix& M);

/** Given a r x nd block matrix composed of n (r x d) blocks that are elements
 * of the Stiefel manifold St(r,d), this function projects the constituent
 * blocks onto SO(d) */
Matrix round_solution(const Matrix& Y, unsigned int d);

/** The SE-Sync problem has infinite solutions equivalent up to an orthogonal
 * transformation. This implies there are two non-connected components,
 * so we need to check if the solution has correctly oriented blocks.
*/
//bool is_solution_reflected(const CartanSyncVariable& x);
bool is_solution_reflected(CartanSyncVariable& x);

/** Given an element of a product of Stiefel manifolds, this function sets the
 * passed matrix Y to the corresponding (extrinsic) block matrix representation
 * of that element (see equations (36) and (37) in the  tech report).  Note that
 * the passed matrix Y must be initialized and of the correct dimensions! */
void StiefelProd2Mat(const ROPTLIB::ProductElement& product_element,
    Matrix& Y);

/** Given an Eigen matrix Y of size r x nd whose (rxd) blocks are elements of
 * the Stiefel manifold St(r,d)^n, this function sets the corresponding
 * ProductElement.  Note that the passed matrix Y must be stored in COLUMN MAJOR
 * ORDER! */
void Mat2StiefelProd(const Eigen::MatrixXd& Y, ROPTLIB::ProductElement& product_element);

/** Given an element of a product of Cartan manifolds, this function sets the
 * passed matrix Y to the corresponding (extrinsic) block matrix representation
 * of that element (see equations (36) and (37) in the  tech report).  Note that
 * the passed matrix Y must be initialized and of the correct dimensions! */
void CartanProd2Mat(const ROPTLIB::ProductElement& product_element,
                    Matrix& Y);

/** Given an Eigen matrix Y of size r x n(d+1) whose (rx(d+1)) blocks are
 * elements of the Cartan manifold (St(r,d) x Re^r)^n, this function sets
 * the corresponding ProductElement.
 * Note that the passed matrix Y must be stored in COLUMN MAJOR ORDER! */
void Mat2CartanProd(const Eigen::MatrixXd& Y,
                    ROPTLIB::ProductElement& product_element);

}

#endif // _SESYNC_UTILS_H_
