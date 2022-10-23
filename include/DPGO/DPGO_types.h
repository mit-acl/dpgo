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
#include <Eigen/CholmodSupport>
#include <SolversTR.h>
#include <RTRNewton.h>
#include <map>
#include <memory>
#include <tuple>

namespace DPGO {

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;
typedef Eigen::CholmodDecomposition<SparseMatrix> CholmodSolver;
typedef std::shared_ptr<CholmodSolver> CholmodSolverPtr;

/**
 * @brief Algorithms for initialize PGO
 */
enum class InitializationMethod {
  Odometry,
  Chordal,
  GNC_TLS
};

std::string InitializationMethodToString(InitializationMethod method);

/**
 * @brief Parameter settings for Riemannian optimization
 */
class ROptParameters {
 public:

  enum class ROptMethod {
    // Riemannian Trust-Region (RTRNewton in ROPTLIB)
    RTR,
    // Riemannian gradient descent (RSD in ROPTLIB)
    RGD
  };
  ROptParameters() :
      method(ROptMethod::RTR),
      verbose(false),
      gradnorm_tol(1e-2),
      RGD_stepsize(1e-3),
      RTR_iterations(3),
      RTR_tCG_iterations(50),
      RTR_initial_radius(100) {}

  ROptMethod method;
  bool verbose;
  double gradnorm_tol;
  double RGD_stepsize;
  int RTR_iterations;
  int RTR_tCG_iterations; // Maximum number of tCG iterations
  double RTR_initial_radius;

  static std::string ROptMethodToString(ROptMethod method);

  inline friend std::ostream &operator<<(
      std::ostream &os, const ROptParameters &params) {
    os << "Riemannian optimization parameters: " << std::endl;
    os << "Method: " << ROptMethodToString(params.method) << std::endl;
    os << "Gradient norm tol: " << params.gradnorm_tol << std::endl;
    os << "RGD stepsize: " << params.RGD_stepsize << std::endl;
    os << "RTR iterations: " << params.RTR_iterations << std::endl;
    os << "RTR tCG iterations: " << params.RTR_tCG_iterations << std::endl;
    os << "RTR initial radius: " << params.RTR_initial_radius << std::endl;
    return os;
  }
};

/**
        Output statistics of Riemannian optimization
*/
struct ROPTResult {
  ROPTResult(bool suc = false, double f0 = 0, double gn0 = 0, double fStar = 0, double gnStar = 0, double ms = 0)
      : success(suc),
        fInit(f0),
        gradNormInit(gn0),
        fOpt(fStar),
        gradNormOpt(gnStar),
        elapsedMs(ms) {}

  bool success;           // Is the optimization successful
  double fInit;           // Objective value before optimization
  double gradNormInit;    // Gradient norm before optimization
  double fOpt;            // Objective value after optimization
  double gradNormOpt;     // Gradient norm after optimization
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
  explicit PoseID(unsigned int rid = 0, unsigned int fid = 0) : robot_id(rid), frame_id(fid) {}
};
// Comparator for PoseID
struct ComparePoseID {
  bool operator()(const PoseID &a, const PoseID &b) const {
    auto pa = std::make_pair(a.robot_id, a.frame_id);
    auto pb = std::make_pair(b.robot_id, b.frame_id);
    return pa < pb;
  }
};

}  // namespace DPGO

#endif
