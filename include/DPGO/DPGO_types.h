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
#include <boost/functional/hash.hpp>

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
      RGD_use_preconditioner(true),
      RTR_iterations(3),
      RTR_tCG_iterations(50),
      RTR_initial_radius(100) {}

  ROptMethod method;
  bool verbose;
  double gradnorm_tol;
  double RGD_stepsize;
  bool RGD_use_preconditioner; 
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
    os << "RGD use preconditioner: " << params.RGD_use_preconditioner << std::endl;
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

// Each pose is uniquely determined by the robot ID and frame ID
class PoseID {
 public:
  unsigned int robot_id;  // robot ID
  unsigned int frame_id;  // frame ID
  explicit PoseID(unsigned int rid = 0, unsigned int fid = 0) : robot_id(rid), frame_id(fid) {}
  PoseID(const PoseID &other) : robot_id(other.robot_id), frame_id(other.frame_id) {}
  bool operator==(const PoseID &other) const
  { return (robot_id == other.robot_id
            && frame_id == other.frame_id);
  }
};
// Comparator for PoseID
struct ComparePoseID {
  bool operator()(const PoseID &a, const PoseID &b) const {
    auto pa = std::make_pair(a.robot_id, a.frame_id);
    auto pb = std::make_pair(b.robot_id, b.frame_id);
    return pa < pb;
  }
};

// Edge measurement (edge) is uniquely determined by an ordered pair of poses
class EdgeID {
 public:
  PoseID src_pose_id;
  PoseID dst_pose_id;
  EdgeID(const PoseID &src_id, const PoseID &dst_id)
      : src_pose_id(src_id), dst_pose_id(dst_id) {}
  bool operator==(const EdgeID &other) const
  { return (src_pose_id == other.src_pose_id
            && dst_pose_id == other.dst_pose_id);
  }
  bool isOdometry() const {
    return (src_pose_id.robot_id == dst_pose_id.robot_id && 
            src_pose_id.frame_id + 1 == dst_pose_id.frame_id);
  }
  bool isPrivateLoopClosure() const {
    return (src_pose_id.robot_id == dst_pose_id.robot_id && 
            src_pose_id.frame_id + 1 != dst_pose_id.frame_id);
  }
  bool isSharedLoopClosure() const {
    return src_pose_id.robot_id != dst_pose_id.robot_id;
  }
};
// Comparator for EdgeID
struct CompareEdgeID {
  bool operator()(const EdgeID &a, const EdgeID &b) const {
    // Treat edge ID as an ordered tuple
    const auto ta = std::make_tuple(a.src_pose_id.robot_id,
                                    a.dst_pose_id.robot_id,
                                    a.src_pose_id.frame_id,
                                    a.dst_pose_id.frame_id);
    const auto tb = std::make_tuple(b.src_pose_id.robot_id,
                                    b.dst_pose_id.robot_id,
                                    b.src_pose_id.frame_id,
                                    b.dst_pose_id.frame_id);
    return ta < tb;
  }
};
// Hasher for EdgeID
struct HashEdgeID
{
  std::size_t operator()(const EdgeID& edge_id) const
  {
      // Reference: 
      // https://stackoverflow.com/questions/17016175/c-unordered-map-using-a-custom-class-type-as-the-key 
      using boost::hash_value;
      using boost::hash_combine;

      // Start with a hash value of 0    .
      std::size_t seed = 0;

      // Modify 'seed' by XORing and bit-shifting in
      // one member of 'Key' after the other:
      hash_combine(seed, hash_value(edge_id.src_pose_id.robot_id));
      hash_combine(seed, hash_value(edge_id.dst_pose_id.robot_id));
      hash_combine(seed, hash_value(edge_id.src_pose_id.frame_id));
      hash_combine(seed, hash_value(edge_id.dst_pose_id.frame_id));

      // Return the result.
      return seed;
  }
};
}  // namespace DPGO

#endif
