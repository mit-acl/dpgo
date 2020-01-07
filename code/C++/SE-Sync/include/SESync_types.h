#ifndef SESYNC_TYPES_H
#define SESYNC_TYPES_H

#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace SESync {

/** Some useful typedefs for the SE-Sync library */

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;

/** We use row-major storage order to take advantage of fast (sparse-matrix) * (dense-vector) multiplications when OpenMP is available */
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;

/** Algorithm types */
enum class AlgType {
  StiefelSync,
  CartanSync
};

} // namespace SESync

#endif // SESYNC_TYPES_H
