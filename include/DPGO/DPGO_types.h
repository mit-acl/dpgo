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
#include <SolversTR.h>
#include <RTRNewton.h>

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
  ROPTResult(bool suc = false, double f0 = 0, double gn0 = 0, double fStar = 0, double gnStar = 0,
             double relchange = 0, double ms = 0)
      : success(suc),
        fInit(f0),
        gradNormInit(gn0),
        fOpt(fStar),
        gradNormOpt(gnStar),
        relativeChange(relchange),
        elapsedMs(ms) {}

  bool success;           // Is the optimization successful
  double fInit;           // Objective value before optimization
  double gradNormInit;    // Gradient norm before optimization
  double fOpt;            // Objective value after optimization
  double gradNormOpt;     // Gradient norm after optimization
  double relativeChange;  // Relative change in solution
  double elapsedMs;       // elapsed time in milliseconds
  ROPTLIB::tCGstatusSet tCGStatus;  // status of truncated conjugate gradient (only used by trust region solver)
};

// In distributed PGO, each pose is uniquely determined by the robot ID and pose ID
// typedef std::pair<unsigned, unsigned> PoseID;

// Each pose is uniquely determined by the robot ID and frame ID
class PoseID {
 public:
  unsigned int robot_id;  // robot ID
  unsigned int frame_id;  // frame ID
  explicit PoseID(unsigned int rid = 0, unsigned int fid = 0): robot_id(rid), frame_id(fid) {}
};
// Comparator for PoseID
struct ComparePoseID {
  bool operator()(const PoseID& a, const PoseID& b) const {
    auto pa = std::make_pair(a.robot_id, a.frame_id);
    auto pb = std::make_pair(b.robot_id, b.frame_id);
    return pa < pb;
  }
};

}  // namespace DPGO

#endif
