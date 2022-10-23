/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGO_INCLUDE_DPGO_PGOSOLVER_H_
#define DPGO_INCLUDE_DPGO_PGOSOLVER_H_

#include <DPGO/DPGO_utils.h>
#include <DPGO/DPGO_robust.h>
#include <DPGO/manifold/Poses.h>

namespace DPGO {
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

/**
 * @brief Initialize local trajectory estimate from chordal relaxation
 * @param measurements
 * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
 */
PoseArray chordalInitialization(const std::vector<RelativeSEMeasurement> &measurements);

/**
 * @brief Initialize local trajectory estimate from odometry
 * @param odometry A vector of odometry measurement
 * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
 */
PoseArray odometryInitialization(const std::vector<RelativeSEMeasurement> &odometry);

/**
 * @brief Perform single-robot pose graph optimization using the L2 cost function
 * @param measurements
 * @param params
 * @param T0
 * @return
 */
PoseArray solvePGO(const std::vector<RelativeSEMeasurement> &measurements,
                   const ROptParameters &params,
                   const PoseArray *T0 = nullptr);

struct solveRobustPGOParams {
 public:
  ROptParameters opt_params;
  RobustCostParameters robust_params;
  bool verbose;
  solveRobustPGOParams() :
      opt_params(),
      robust_params(RobustCostParameters::Type::GNC_TLS),
      verbose(true) {}
};

/**
 * @brief Perform single-robot pose graph optimization using graduated non-convexity (GNC)
 * @param mutable_measurements
 * @param params
 * @param T0
 * @return
 */
PoseArray solveRobustPGO(std::vector<RelativeSEMeasurement> &mutable_measurements,
                         const solveRobustPGOParams &params,
                         const PoseArray *T0 = nullptr);

}

#endif //DPGO_INCLUDE_DPGO_PGOSOLVER_H_
