/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGO_TYPES_H
#define DPGO_TYPES_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <map>
#include <tuple>

namespace DPGO {

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;

/**
        Riemannian optimization algorithms from ROPTLIB
        to be used as solver.
*/
enum ROPTALG {
  // Riemannian Trust-Region (RTRNewton in ROPTLIB)
  RTR,

  // Riemannian gradient descent (RSD in ROPTLIB)
  RGD
};

/**
        Output statistics of Riemannian optimization
*/
struct ROPTResult {
  ROPTResult() {}
  ROPTResult(bool suc) : success(suc) {}
  ROPTResult(bool suc, double f0, double gn0, double fStar, double gnStar,
             double ms)
      : success(suc),
        fInit(f0),
        gradNormInit(gn0),
        fOpt(fStar),
        gradNormOpt(gnStar),
        elapsedMs(ms) {}

  bool success;         // Is the optimization successful
  double fInit;         // Objective value before optimization
  double gradNormInit;  // Gradient norm before optimization
  double fOpt;          // Objective value after optimization
  double gradNormOpt;   // Gradient norm after optimization
  double elapsedMs;     // elapsed time in milliseconds
};

// In distributed PGO, each pose is uniquely determined by the robot ID and pose
// ID
typedef std::pair<unsigned, unsigned> PoseID;

// Implement a dictionary for easy access of pose value by PoseID
typedef std::map<PoseID, Matrix, std::less<PoseID>,
                 Eigen::aligned_allocator<std::pair<const PoseID, Matrix>>>
    PoseDict;

}  // namespace DPGO

#endif
