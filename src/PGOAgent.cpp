/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticOptimizer.h>

#include <Eigen/CholmodSupport>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include "StieVariable.h"

using std::lock_guard;
using std::unique_lock;
using std::set;
using std::thread;
using std::vector;

namespace DPGO {

PGOAgent::PGOAgent(unsigned ID, const PGOAgentParameters &params)
    : mID(ID), d(params.d), r(params.r), n(1),
      mParams(params), mState(PGOAgentState::WAIT_FOR_DATA),
      mStatus(ID, mState, 0, 0, false, 0),
      mRobustCost(params.robustCostType, params.robustCostParams),
      mProblemPtr(nullptr),
      mInstanceNumber(0), mIterationNumber(0), mNumPosesReceived(0),
      mLogger(params.logDirectory) {
  if (mParams.verbose) {
    std::cout << "Initializing PGO agent..." << std::endl;
    std::cout << params << std::endl;
  }

  // Initialize X
  X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);
  if (mID == 0) setLiftingMatrix(fixedStiefelVariable(d, r));
  resetTeamStatus();
}

PGOAgent::~PGOAgent() {
  // Make sure that optimization thread is not running, before exiting
  endOptimizationLoop();
}

void PGOAgent::setX(const Matrix &Xin) {
  lock_guard<mutex> lock(mPosesMutex);
  assert(mState != PGOAgentState::WAIT_FOR_DATA);
  assert(Xin.rows() == relaxation_rank());
  assert(Xin.cols() == (dimension() + 1) * num_poses());
  mState = PGOAgentState::INITIALIZED;
  X = Xin;
  if (mParams.acceleration) {
    initializeAcceleration();
  }
  if (mParams.verbose) {
    printf("Robot %u resets trajectory estimates. New trajectory length = %u\n", getID(), num_poses());
  }
}

bool PGOAgent::getX(Matrix &Mout) {
  lock_guard<mutex> lock(mPosesMutex);
  Mout = X;
  return true;
}

bool PGOAgent::getSharedPose(unsigned int index, Matrix &Mout) {
  if (mState != PGOAgentState::INITIALIZED)
    return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (index >= num_poses()) return false;
  Mout = X.block(0, index * (d + 1), r, d + 1);
  return true;
}

bool PGOAgent::getAuxSharedPose(unsigned int index, Matrix &Mout) {
  assert(mParams.acceleration);
  if (mState != PGOAgentState::INITIALIZED)
    return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (index >= num_poses()) return false;
  Mout = Y.block(0, index * (d + 1), r, d + 1);
  return true;
}

bool PGOAgent::getSharedPoseDict(PoseDict &map) {
  if (mState != PGOAgentState::INITIALIZED)
    return false;
  map.clear();
  lock_guard<mutex> lock(mPosesMutex);
  for (const auto &mSharedPose: localSharedPoseIDs) {
    unsigned idx = std::get<1>(mSharedPose);
    map[mSharedPose] = X.block(0, idx * (d + 1), r, d + 1);
  }
  return true;
}

bool PGOAgent::getAuxSharedPoseDict(PoseDict &map) {
  assert(mParams.acceleration);
  if (mState != PGOAgentState::INITIALIZED)
    return false;
  map.clear();
  lock_guard<mutex> lock(mPosesMutex);
  for (const auto &mSharedPose: localSharedPoseIDs) {
    unsigned idx = std::get<1>(mSharedPose);
    map[mSharedPose] = Y.block(0, idx * (d + 1), r, d + 1);
  }
  return true;
}

void PGOAgent::setLiftingMatrix(const Matrix &M) {
  assert(M.rows() == r);
  assert(M.cols() == d);
  YLift.emplace(M);
}

void PGOAgent::setPoseGraph(
    const std::vector<RelativeSEMeasurement> &inputOdometry,
    const std::vector<RelativeSEMeasurement> &inputPrivateLoopClosures,
    const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures,
    const Matrix &TInit) {
  assert(!isOptimizationRunning());
  assert(mState == PGOAgentState::WAIT_FOR_DATA);
  assert(n == 1);

  if (inputOdometry.empty()) return;

  for (const auto &edge: inputOdometry) {
    addOdometry(edge);
  }
  for (const auto &edge: inputPrivateLoopClosures) {
    addPrivateLoopClosure(edge);
  }
  for (const auto &edge: inputSharedLoopClosures) {
    addSharedLoopClosure(edge);
  }

  // Check validity of initial trajectory estimate, if provided
  bool local_init = true;
  unsigned expected_rows = dimension();
  unsigned expected_cols = (dimension() + 1) * num_poses();
  if (TInit.rows() > 0 && TInit.cols() > 0) {
    if (TInit.rows() == expected_rows && TInit.cols() == expected_cols) {
      local_init = false;
    } else {
      local_init = true;
      printf("Error: provided initial trajectory has wrong dimension! "
             "Expect (%u,%u), received (%ld, %ld). Using local initialization. \n",
             expected_rows, expected_cols, TInit.rows(), TInit.cols());
    }
  }

  // Create new optimization problem
  mProblemPtr = new QuadraticProblem(num_poses(), dimension(), relaxation_rank());

  // Robot can construct the quadratic cost matrix now, as it does not depend on neighbor values
  constructQMatrix();

  // Initialize trajectory estimate in an arbitrary frame
  if (!local_init) {
    if (mParams.verbose) printf("Using provided trajectory initialization.\n");
    TLocalInit.emplace(TInit.block(0, 0, expected_rows, expected_cols));
  } else {
    if (mParams.verbose) printf("Using internal trajectory initialization.\n");
    localInitialization();
  }

  // Waiting for initialization in the GLOBAL frame
  mState = PGOAgentState::WAIT_FOR_INITIALIZATION;

  // If I am the first robot or if cross-robot initialization if off,
  // I will consider myself as initialized in the global frame
  if (mID == 0 || !mParams.multirobot_initialization) {
    X = YLift.value() * TLocalInit.value();  // Lift to correct relaxation rank
    XInit.emplace(X);
    mState = PGOAgentState::INITIALIZED;
    if (mParams.acceleration) {
      initializeAcceleration();
    }

    // Save initial trajectory
    if (mParams.logData) {
      mLogger.logTrajectory(dimension(), num_poses(), TLocalInit.value(), "trajectory_initial.csv");
    }
  }
}

void PGOAgent::addOdometry(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  // check that this is a odometry measurement
  assert(factor.r1 == mID);
  assert(factor.r2 == mID);
  assert(factor.p1 + 1 == factor.p2);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);

  // update number of poses
  n = std::max(n, (unsigned) factor.p2 + 1);

  lock_guard<mutex> mLock(mMeasurementsMutex);
  odometry.push_back(factor);
}

void PGOAgent::addPrivateLoopClosure(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  assert(factor.r1 == mID);
  assert(factor.r2 == mID);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);

  // update number of poses
  n = std::max(n, (unsigned) std::max(factor.p1 + 1, factor.p2 + 1));

  lock_guard<mutex> lock(mMeasurementsMutex);
  privateLoopClosures.push_back(factor);
}

void PGOAgent::addSharedLoopClosure(const RelativeSEMeasurement &factor) {
  assert(mState != PGOAgentState::INITIALIZED);
  assert(factor.R.rows() == d && factor.R.cols() == d);
  assert(factor.t.rows() == d && factor.t.cols() == 1);

  if (factor.r1 == mID) {
    assert(factor.r2 != mID);
    n = std::max(n, (unsigned) factor.p1 + 1);
    localSharedPoseIDs.insert(std::make_pair(mID, factor.p1));
    neighborSharedPoseIDs.insert(std::make_pair(factor.r2, factor.p2));
    neighborRobotIDs.insert(factor.r2);
  } else {
    assert(factor.r2 == mID);
    n = std::max(n, (unsigned) factor.p2 + 1);
    localSharedPoseIDs.insert(std::make_pair(mID, factor.p2));
    neighborSharedPoseIDs.insert(std::make_pair(factor.r1, factor.p1));
    neighborRobotIDs.insert(factor.r1);
  }

  lock_guard<mutex> lock(mMeasurementsMutex);
  sharedLoopClosures.push_back(factor);
}

Matrix PGOAgent::computeNeighborTransform(const PoseID &nID, const Matrix &var) {
  assert(YLift);
  assert(var.rows() == r);
  assert(var.cols() == d + 1);

  // Find the corresponding inter-robot loop closure
  RelativeSEMeasurement &m = findSharedLoopClosureWithNeighbor(nID);

  // Notations:
  // world1: world frame before alignment
  // world2: world frame after alignment
  // frame1 : frame associated to my public pose
  // frame2 : frame associated to neighbor's public pose
  Matrix dT = Matrix::Identity(d + 1, d + 1);
  dT.block(0, 0, d, d) = m.R;
  dT.block(0, d, d, 1) = m.t;
  Matrix T_world2_frame2 = Matrix::Identity(d + 1, d + 1);
  T_world2_frame2.block(0, 0, d, d + 1) =
      YLift.value().transpose() *
          var;  // Round the received neighbor pose value back to SE(d)
  Matrix T = TLocalInit.value();
  Matrix T_frame1_frame2 = Matrix::Identity(d + 1, d + 1);
  Matrix T_world1_frame1 = Matrix::Identity(d + 1, d + 1);
  if (m.r1 == nID.first) {
    // Incoming edge
    T_frame1_frame2 = dT.inverse();
    T_world1_frame1.block(0, 0, d, d + 1) =
        T.block(0, m.p2 * (d + 1), d, d + 1);
  } else {
    // Outgoing edge
    T_frame1_frame2 = dT;
    T_world1_frame1.block(0, 0, d, d + 1) =
        T.block(0, m.p1 * (d + 1), d, d + 1);
  }
  Matrix T_world2_frame1 = T_world2_frame2 * T_frame1_frame2.inverse();
  Matrix T_world2_world1 = T_world2_frame1 * T_world1_frame1.inverse();
  checkRotationMatrix(T_world2_world1.block(0, 0, d, d));
  return T_world2_world1;
}

Matrix PGOAgent::computeRobustNeighborTransformTwoStage(unsigned int neighborID, const PoseDict &poseDict) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  for (const auto &it: poseDict) {
    const PoseID nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      const auto T = computeNeighborTransform(nID, var);
      RVec.emplace_back(T.block(0, 0, d, d));
      tVec.emplace_back(T.block(0, d, d, 1));
    }
  }
  int m = (int) RVec.size();
  const Vector kappa = Vector::Ones(m);
  const Vector tau = Vector::Ones(m);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;
  // Perform robust single rotation averaging
  double maxRotationError = angular2ChordalSO3(0.5);  // approximately 30 deg
  robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa, maxRotationError);
  int inlierSize = (int) inlierIndices.size();
  printf("[RobustRelativeTransform] This robot %u, neighbor %u: finds %i inliers out of %i measurements.\n",
         getID(),
         neighborID,
         inlierSize,
         m);
  if (inlierSize == 0) {
    throw std::runtime_error("Robust single rotation averaging returns empty inlier set!");
  }
  // Perform single translation averaging on the inlier set
  std::vector<Vector> tVecInliers;
  for (const auto index: inlierIndices) {
    tVecInliers.emplace_back(tVec[index]);
  }
  singleTranslationAveraging(tOpt, tVecInliers);
  // Return transformation as matrix
  Matrix TOpt = Matrix::Identity(dimension() + 1, dimension() + 1);
  TOpt.block(0, 0, d, d) = ROpt;
  TOpt.block(0, d, d, 1) = tOpt;
  return TOpt;
}

Matrix PGOAgent::computeRobustNeighborTransform(unsigned int neighborID, const PoseDict &poseDict) {
  std::vector<Matrix> RVec;
  std::vector<Vector> tVec;
  for (const auto &it: poseDict) {
    const PoseID nID = it.first;
    const auto var = it.second;
    if (neighborSharedPoseIDs.find(nID) != neighborSharedPoseIDs.end()) {
      const auto T = computeNeighborTransform(nID, var);
      RVec.emplace_back(T.block(0, 0, d, d));
      tVec.emplace_back(T.block(0, d, d, 1));
    }
  }
  int m = (int) RVec.size();
  const Vector kappa = 1.82 * Vector::Ones(m);  // rotation stddev approximately 30 degree
  const Vector tau = 0.01 * Vector::Ones(m);  // translation stddev 10 m
  const double cbar = RobustCost::computeErrorThresholdAtQuantile(0.9, 3);
  Matrix ROpt;
  Vector tOpt;
  std::vector<size_t> inlierIndices;
  robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa, tau, cbar);
  int inlierSize = (int) inlierIndices.size();
  printf("[RobustRelativeTransform] This robot %u, neighbor %u: finds %i inliers out of %i measurements.\n",
         getID(),
         neighborID,
         inlierSize,
         m);
  if (inlierSize == 0) {
    throw std::runtime_error("Robust single pose averaging returns empty inlier set!");
  }
  // Return transformation as matrix
  Matrix TOpt = Matrix::Identity(dimension() + 1, dimension() + 1);
  TOpt.block(0, 0, d, d) = ROpt;
  TOpt.block(0, d, d, 1) = tOpt;
  return TOpt;
}

void PGOAgent::initializeInGlobalFrame(unsigned neighborID, const PoseDict &poseDict) {
  // Require the lifting matrix to initialize
  assert(YLift);

  // Halt optimization
  bool optimizationHalted = false;
  if (isOptimizationRunning()) {
    if (mParams.verbose)
      printf("Robot %u halting optimization thread...\n", getID());
    optimizationHalted = true;
    endOptimizationLoop();
  }

  // Halt insertion of new poses
  lock_guard<mutex> tLock(mPosesMutex);

  // Halt insertion of new measurements
  lock_guard<mutex> mLock(mMeasurementsMutex);

  // Clear cache
  lock_guard<mutex> nLock(mNeighborPosesMutex);
  neighborPoseDict.clear();
  neighborAuxPoseDict.clear();

  // Compute relative transform to neighbor's frame of reference
  Matrix T_world2_world1;
  try {
    T_world2_world1 = computeRobustNeighborTransformTwoStage(neighborID, poseDict);
  } catch (const std::runtime_error& e) {
    printf("Robust initialization is not successful! Abort and wait to try again...\n");
    return;
  }

  // Apply global transformation to local trajectory estimate
  Matrix T = TLocalInit.value();
  Matrix T_world1_frame = Matrix::Identity(d + 1, d + 1);
  Matrix T_world2_frame = Matrix::Identity(d + 1, d + 1);
  for (size_t i = 0; i < num_poses(); ++i) {
    T_world1_frame.block(0, 0, d, d + 1) =
        T.block(0, i * (d + 1), d, d + 1);
    T_world2_frame = T_world2_world1 * T_world1_frame;
    T.block(0, i * (d + 1), d, d + 1) =
        T_world2_frame.block(0, 0, d, d + 1);
  }

  // Lift back to correct relaxation rank
  X = YLift.value() * T;
  XInit.emplace(X);

  // Mark this agent as initialized
  mState = PGOAgentState::INITIALIZED;

  // Initialize auxiliary variables
  if (mParams.acceleration) {
    initializeAcceleration();
  }

  // Log initial trajectory
  if (mParams.logData) {
    mLogger.logTrajectory(dimension(), num_poses(), T, "trajectory_initial.csv");
  }

  if (optimizationHalted) startOptimizationLoop(mRate);
}

void PGOAgent::updateNeighborPoses(unsigned neighborID, const PoseDict &poseDict) {
  assert(neighborID != mID);
  // Initialize this robot in the global frame, if not initialized
  const auto neighborState = getNeighborStatus(neighborID).state;
  if (mState == PGOAgentState::WAIT_FOR_INITIALIZATION && neighborState == PGOAgentState::INITIALIZED) {
    initializeInGlobalFrame(neighborID, poseDict);
  }
  // Save neighbor public poses in local cache
  for (const auto &it: poseDict) {
    const auto nID = it.first;
    const auto var = it.second;
    assert(nID.first == neighborID);
    assert(var.rows() == r);
    assert(var.cols() == d + 1);
    mNumPosesReceived++;
    if (neighborSharedPoseIDs.find(nID) == neighborSharedPoseIDs.end())
      continue;
    // Only save poses from neighbors if this agent is initialized
    // and if the sending agent is also initialized
    if (mState == PGOAgentState::INITIALIZED && neighborState == PGOAgentState::INITIALIZED) {
      lock_guard<mutex> lock(mNeighborPosesMutex);
      neighborPoseDict[nID] = var;
    }
  }
}

void PGOAgent::updateAuxNeighborPoses(unsigned neighborID, const PoseDict &poseDict) {
  assert(mParams.acceleration);
  assert(neighborID != mID);
  for (const auto &it: poseDict) {
    const auto nID = it.first;
    const auto var = it.second;
    assert(nID.first == neighborID);
    assert(var.rows() == r);
    assert(var.cols() == d + 1);
    mNumPosesReceived++;
    if (neighborSharedPoseIDs.find(nID) == neighborSharedPoseIDs.end())
      continue;
    // Only save poses from neighbors if this agent is initialized
    // and if the sending agent is also initialized
    if (mState == PGOAgentState::INITIALIZED && getNeighborStatus(neighborID).state == PGOAgentState::INITIALIZED) {
      lock_guard<mutex> lock(mNeighborPosesMutex);
      neighborAuxPoseDict[nID] = var;
    }
  }
}

bool PGOAgent::getTrajectoryInLocalFrame(Matrix &Trajectory) {
  if (mState != PGOAgentState::INITIALIZED) {
    return false;
  }
  lock_guard<mutex> lock(mPosesMutex);

  Matrix T = X.block(0, 0, r, d).transpose() * X;
  Matrix t0 = T.block(0, d, d, 1);

  for (unsigned i = 0; i < n; ++i) {
    T.block(0, i * (d + 1), d, d) =
        projectToRotationGroup(T.block(0, i * (d + 1), d, d));
    T.block(0, i * (d + 1) + d, d, 1) = T.block(0, i * (d + 1) + d, d, 1) - t0;
  }

  Trajectory = T;
  return true;
}

bool PGOAgent::getTrajectoryInGlobalFrame(Matrix &Trajectory) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);

  Matrix T = globalAnchor.value().block(0, 0, r, d).transpose() * X;
  Matrix t0 = globalAnchor.value().block(0, 0, r, d).transpose() *
      globalAnchor.value().block(0, d, r, 1);

  for (unsigned i = 0; i < n; ++i) {
    T.block(0, i * (d + 1), d, d) =
        projectToRotationGroup(T.block(0, i * (d + 1), d, d));
    T.block(0, i * (d + 1) + d, d, 1) = T.block(0, i * (d + 1) + d, d, 1) - t0;
  }

  Trajectory = T;
  return true;
}

bool PGOAgent::getPoseInGlobalFrame(unsigned int poseID, Matrix &T) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mPosesMutex);
  if (poseID < 0 || poseID >= num_poses()) return false;
  Matrix Ya = globalAnchor.value().block(0, 0, r, d);
  Matrix pa = globalAnchor.value().block(0, d, r, 1);
  Matrix t0 = Ya.transpose() * pa;
  Matrix Xi = X.block(0, poseID * (d + 1), r, d + 1);
  Matrix Ti = Ya.transpose() * Xi;
  Ti.block(0, d, d, 1) -= t0;
  assert(Ti.rows() == d);
  assert(Ti.cols() == d + 1);
  T = Ti;
  return true;
}

bool PGOAgent::getNeighborPoseInGlobalFrame(unsigned int neighborID, unsigned int poseID, Matrix &T) {
  if (!globalAnchor) return false;
  assert(globalAnchor.value().rows() == relaxation_rank());
  assert(globalAnchor.value().cols() == dimension() + 1);
  if (mState != PGOAgentState::INITIALIZED) return false;
  lock_guard<mutex> lock(mNeighborPosesMutex);
  PoseID nID = std::make_pair(neighborID, poseID);
  if (neighborPoseDict.find(nID) != neighborPoseDict.end()) {
    Matrix Ya = globalAnchor.value().block(0, 0, r, d);
    Matrix pa = globalAnchor.value().block(0, d, r, 1);
    Matrix t0 = Ya.transpose() * pa;
    Matrix Xi = neighborPoseDict.at(nID);
    assert(Xi.rows() == r);
    assert(Xi.cols() == d + 1);
    Matrix Ti = Ya.transpose() * Xi;
    Ti.block(0, d, d, 1) -= t0;
    assert(Ti.rows() == d);
    assert(Ti.cols() == d + 1);
    T = Ti;
    return true;
  }
  return false;
}

std::vector<unsigned> PGOAgent::getNeighborPublicPoses(
    const unsigned &neighborID) const {
  // Check that neighborID is indeed a neighbor of this agent
  assert(neighborRobotIDs.find(neighborID) != neighborRobotIDs.end());
  std::vector<unsigned> poseIndices;
  for (PoseID pair: neighborSharedPoseIDs) {
    if (pair.first == neighborID) {
      poseIndices.push_back(pair.second);
    }
  }
  return poseIndices;
}

std::vector<unsigned> PGOAgent::getNeighbors() const {
  std::vector<unsigned> v(neighborRobotIDs.size());
  std::copy(neighborRobotIDs.begin(), neighborRobotIDs.end(), v.begin());
  return v;
}

void PGOAgent::reset() {
  // Terminate optimization thread if running
  endOptimizationLoop();

  if (mParams.logData) {
    // Save measurements (including final weights)
    std::vector<RelativeSEMeasurement> measurements = odometry;
    measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());
    measurements.insert(measurements.end(), sharedLoopClosures.begin(), sharedLoopClosures.end());
    mLogger.logMeasurements(measurements, "measurements.csv");

    // Save trajectory estimates after rounding
    Matrix T;
    if (getTrajectoryInGlobalFrame(T)) {
      mLogger.logTrajectory(dimension(), num_poses(), T, "trajectory_optimized.csv");
      std::cout << "Saved optimized trajectory to " << mParams.logDirectory << std::endl;
    }

    // Save solution before rounding
    writeMatrixToFile(X, mParams.logDirectory + "X.txt");
  }

  mInstanceNumber++;
  mIterationNumber = 0;
  mNumPosesReceived = 0;

  // Assume that the old lifting matrix can still be used
  mState = PGOAgentState::WAIT_FOR_DATA;
  mStatus = PGOAgentStatus(getID(), mState, mInstanceNumber, mIterationNumber, false, 0);

  odometry.clear();
  privateLoopClosures.clear();
  sharedLoopClosures.clear();

  neighborPoseDict.clear();
  neighborAuxPoseDict.clear();
  localSharedPoseIDs.clear();
  neighborSharedPoseIDs.clear();
  neighborRobotIDs.clear();
  resetTeamStatus();

  if (mProblemPtr) {
    delete mProblemPtr;
    mProblemPtr = nullptr;
  }
  mRobustCost.reset();
  globalAnchor.reset();
  TLocalInit.reset();
  XInit.reset();

  mOptimizationRequested = false;
  mPublishPublicPosesRequested = false;
  mPublishWeightsRequested = false;

  n = 1;
  X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);
}

void PGOAgent::iterate(bool doOptimization) {
  mIterationNumber++;

  // Save early stopped solution
  if (mIterationNumber == 50 && mParams.logData) {
    Matrix T;
    if (getTrajectoryInGlobalFrame(T)) {
      mLogger.logTrajectory(dimension(), num_poses(), T, "trajectory_early_stop.csv");
    }
  }

  // Update measurement weights (GNC)
  if (shouldUpdateLoopClosureWeights()) {
    updateLoopClosuresWeights();
    mRobustCost.update();
    // If warm start is disabled, reset trajectory estimate to initial guess
    if (!mParams.robustOptWarmStart) {
      assert(XInit);
      X = XInit.value();
      printf("Warm start is disabled. Robot %u resets trajectory estimates.\n", getID());
    }
    // Reset acceleration
    if (mParams.acceleration) {
      initializeAcceleration();
    }

  }

  // Perform iteration
  if (mState == PGOAgentState::INITIALIZED) {
    // Save current iterate
    XPrev = X;

    // lock pose update
    unique_lock<mutex> tLock(mPosesMutex);

    // lock measurements
    unique_lock<mutex> mLock(mMeasurementsMutex);

    // lock neighbor pose update
    unique_lock<mutex> lock(mNeighborPosesMutex);

    bool success;
    if (mParams.acceleration) {
      updateGamma();
      updateAlpha();
      updateY();
      success = updateX(doOptimization, true);
      updateV();
      // Check restart condition
      if (shouldRestart()) {
        restartNesterovAcceleration(doOptimization);
      }
      mPublishPublicPosesRequested = true;
    } else {
      success = updateX(doOptimization, false);
      if (doOptimization) {
        mPublishPublicPosesRequested = true;
      }
    }

    if (doOptimization) {
      mStatus.agentID = getID();
      mStatus.state = mState;
      mStatus.instanceNumber = instance_number();
      mStatus.iterationNumber = iteration_number();
      mStatus.relativeChange = sqrt((X - XPrev).squaredNorm() / num_poses());
      // Check local termination condition
      bool readyToTerminate = true;
      if (!success) readyToTerminate = false;
      if (mStatus.relativeChange > mParams.relChangeTol) readyToTerminate = false;
      double ratio = computeConvergedLoopClosureRatio();
      if (ratio < mParams.robustOptMinConvergenceRatio) readyToTerminate = false;
      mStatus.readyToTerminate = readyToTerminate;
    }
  }
}

void PGOAgent::constructQMatrix() {
  vector<RelativeSEMeasurement> privateMeasurements = odometry;
  privateMeasurements.insert(privateMeasurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());

  // Initialize Q with private measurements
  SparseMatrix Q = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d + 1, d + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d + 1, d + 1);

  // Go through shared loop closures
  for (const auto &m: sharedLoopClosures) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Set aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    if (m.r1 == mID) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      assert(m.r2 != mID);

      // Modify quadratic cost
      size_t idx = m.p1;

      Matrix W = T * Omega * T.transpose();

      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(idx * (d + 1) + row, idx * (d + 1) + col) += W(row, col);
        }
      }

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      assert(m.r2 == mID);

      // Modify quadratic cost
      size_t idx = m.p2;

      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < d + 1; ++row) {
          Q.coeffRef(idx * (d + 1) + row, idx * (d + 1) + col) +=
              Omega(row, col);
        }
      }
    }
  }

  assert(mProblemPtr);
  mProblemPtr->setQ(Q);
}

bool PGOAgent::constructGMatrix(const PoseDict &poseDict) {
  SparseMatrix G(relaxation_rank(), (dimension() + 1) * num_poses());

  for (const auto &m: sharedLoopClosures) {
    // Construct relative SE matrix in homogeneous form
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Construct aggregate weight matrix
    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d, d) = m.weight * m.tau;

    if (m.r1 == mID) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      assert(m.r2 != mID);

      // Read neighbor's pose
      const PoseID nID = std::make_pair(m.r2, m.p2);
      auto KVpair = poseDict.find(nID);
      if (KVpair == poseDict.end()) {
        if (mParams.verbose) {
          printf("constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
                 getID(), nID.first, nID.second);
        }
        return false;
      }
      Matrix Xj = KVpair->second;

      size_t idx = m.p1;

      // Modify linear cost
      Matrix L = -Xj * Omega * T.transpose();
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, idx * (d + 1) + col) += L(row, col);
        }
      }

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      assert(m.r2 == mID);

      // Read neighbor's pose
      const PoseID nID = std::make_pair(m.r1, m.p1);
      auto KVpair = poseDict.find(nID);
      if (KVpair == poseDict.end()) {
        if (mParams.verbose) {
          printf("constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
                 getID(), nID.first, nID.second);
        }
        return false;
      }
      Matrix Xi = KVpair->second;

      size_t idx = m.p2;

      // Modify linear cost
      Matrix L = -Xi * T * Omega;
      for (size_t col = 0; col < d + 1; ++col) {
        for (size_t row = 0; row < r; ++row) {
          G.coeffRef(row, idx * (d + 1) + col) += L(row, col);
        }
      }
    }
  }

  assert(mProblemPtr);
  mProblemPtr->setG(G);
  return true;
}

void PGOAgent::startOptimizationLoop(double freq) {
  // Asynchronous updates currently restricted to non-accelerated updates
  assert(!mParams.acceleration);

  if (isOptimizationRunning()) {
    if (mParams.verbose)
      printf("startOptimizationLoop: optimization thread already running! \n");
    return;
  }

  mRate = freq;

  mOptimizationThread = new thread(&PGOAgent::runOptimizationLoop, this);
}

void PGOAgent::runOptimizationLoop() {
  if (mParams.verbose)
    printf("Robot %u optimization thread running at %f Hz.\n", getID(), mRate);

  // Create exponential distribution with the desired rate
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::exponential_distribution<double> ExponentialDistribution(mRate);

  while (true) {
    double sleepUs =
        1e6 * ExponentialDistribution(rng);  // sleeping time in microsecond

    usleep(sleepUs);

    iterate(true);

    // Check if finish requested
    if (mEndLoopRequested) {
      break;
    }
  }
}

void PGOAgent::endOptimizationLoop() {
  if (!isOptimizationRunning()) return;

  mEndLoopRequested = true;

  // wait for thread to finish
  mOptimizationThread->join();

  delete mOptimizationThread;

  mOptimizationThread = nullptr;

  mEndLoopRequested = false;  // reset request flag

  if (mParams.verbose)
    printf("Robot %u optimization thread exited. \n", getID());
}

bool PGOAgent::isOptimizationRunning() {
  return mOptimizationThread != nullptr;
}

RelativeSEMeasurement &PGOAgent::findSharedLoopClosureWithNeighbor(const PoseID &nID) {
  const unsigned neighborID = nID.first;
  const unsigned neighborPose = nID.second;
  for (auto &m: sharedLoopClosures) {
    if ((m.r1 == neighborID && m.p1 == neighborPose) ||
        (m.r2 == neighborID && m.p2 == neighborPose)) {
      return m;
    }
  }

  // The desired measurement is not found. Throw a runtime error.
  throw std::runtime_error("Cannot find shared loop closure with neighbor.");
}

RelativeSEMeasurement &PGOAgent::findSharedLoopClosure(const PoseID &srcID, const PoseID &dstID) {
  for (auto &m: sharedLoopClosures) {
    if (m.r1 == srcID.first && m.p1 == srcID.second && dstID.first == m.r2 && dstID.second == m.p2) {
      return m;
    }
  }

  // The desired measurement is not found. Throw a runtime error.
  throw std::runtime_error("Cannot find specified shared loop closure.");
}

void PGOAgent::localInitialization() {
  std::vector<RelativeSEMeasurement> measurements = odometry;
  measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());

  Matrix T0;
  if (mParams.robustCostType == RobustCostType::L2) {
    T0 = chordalInitialization(dimension(), num_poses(), measurements);
  } else {
    // In robust mode, we do not trust the loop closures and hence initialize from odometry
    T0 = odometryInitialization(dimension(), num_poses(), odometry);
  }

  assert(T0.rows() == d);
  assert(T0.cols() == (d + 1) * n);
  TLocalInit.emplace(T0);
}

Matrix PGOAgent::localPoseGraphOptimization() {
  // Compute initialization if necessary
  if (!TLocalInit)
    localInitialization();

  // Compute connection laplacian
  std::vector<RelativeSEMeasurement> measurements = odometry;
  measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());
  SparseMatrix Q = constructConnectionLaplacianSE(measurements);

  // Form optimization problem
  QuadraticProblem problem(n, d, d);
  problem.setQ(Q);

  // Initialize optimizer object
  QuadraticOptimizer optimizer(&problem);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setTrustRegionInitialRadius(10);
  optimizer.setTrustRegionIterations(10);
  optimizer.setTrustRegionTolerance(1e-1);
  optimizer.setTrustRegionMaxInnerIterations(50);

  // Optimize
  Matrix Topt = optimizer.optimize(TLocalInit.value());
  if (mParams.verbose) printf("Optimization time: %f sec.\n", optimizer.getOptResult().elapsedMs / 1e3);
  return Topt;
}

bool PGOAgent::getLiftingMatrix(Matrix &M) const {
  assert(mID == 0);
  if (YLift.has_value()) {
    M = YLift.value();
    return true;
  }
  return false;
}

void PGOAgent::setGlobalAnchor(const Matrix &M) {
  assert(M.rows() == relaxation_rank());
  assert(M.cols() == dimension() + 1);
  globalAnchor.emplace(M);
}

bool PGOAgent::shouldTerminate() {
  // terminate if reached maximum iterations
  if (iteration_number() > mParams.maxNumIters) {
    printf("Reached maximum iterations.\n");
    return true;
  }

  for (size_t robot = 0; robot < mParams.numRobots; ++robot) {
    PGOAgentStatus robotStatus = mTeamStatus[robot];
    assert(robotStatus.agentID == robot);
    if (robotStatus.state != PGOAgentState::INITIALIZED) {
      return false;
    }
  }

  // Check if all agents are ready to terminate optimization
  for (size_t robot = 0; robot < mParams.numRobots; ++robot) {
    PGOAgentStatus robotStatus = mTeamStatus[robot];
    if (!robotStatus.readyToTerminate) {
      return false;
    }
  }

  return true;
}

bool PGOAgent::shouldRestart() const {
  if (mParams.acceleration) {
    return ((mIterationNumber + 1) % mParams.restartInterval == 0);
  }
  return false;
}

void PGOAgent::restartNesterovAcceleration(bool doOptimization) {
  if (mParams.acceleration && mState == PGOAgentState::INITIALIZED) {
    if (mParams.verbose) {
      printf("Robot %u restarts Nesteorv acceleration.\n", getID());
    }
    X = XPrev;
    updateX(doOptimization, false);
    V = X;
    Y = X;
    gamma = 0;
    alpha = 0;
  }
}

void PGOAgent::initializeAcceleration() {
  assert(mParams.acceleration);
  if (mState == PGOAgentState::INITIALIZED) {
    XPrev = X;
    gamma = 0;
    alpha = 0;
    V = X;
    Y = X;
  }
}

void PGOAgent::updateGamma() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  gamma = (1 + sqrt(1 + 4 * pow(mParams.numRobots, 2) * pow(gamma, 2))) / (2 * mParams.numRobots);
}

void PGOAgent::updateAlpha() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  alpha = 1 / (gamma * mParams.numRobots);
}

void PGOAgent::updateY() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  LiftedSEManifold manifold(relaxation_rank(), dimension(), num_poses());
  Matrix M = (1 - alpha) * X + alpha * V;
  Y = manifold.project(M);
}

void PGOAgent::updateV() {
  assert(mParams.acceleration);
  assert(mState == PGOAgentState::INITIALIZED);
  LiftedSEManifold manifold(relaxation_rank(), dimension(), num_poses());
  Matrix M = V + gamma * (X - Y);
  V = manifold.project(M);
}

bool PGOAgent::updateX(bool doOptimization, bool acceleration) {
  if (!doOptimization) {
    if (acceleration) {
      X = Y;
    }
    return true;
  }

  if (mParams.verbose)
    printf("Robot %u optimize at iteration %u... \n", getID(), iteration_number());

  if (acceleration)
    assert(mParams.acceleration);

  assert(mState == PGOAgentState::INITIALIZED);

  // Update quadratic cost matrix (unless using L2 cost function since it does not change measurement weights)
  if (mParams.robustCostType != RobustCostType::L2) {
    constructQMatrix();
  }

  // Construct linear cost matrix (depending on neighbor robots' poses)
  bool hasG;
  if (acceleration) {
    hasG = constructGMatrix(neighborAuxPoseDict);
  } else {
    hasG = constructGMatrix(neighborPoseDict);
  }

  // Skip update if G matrix is not constructed successfully
  if (!hasG) {
    if (mParams.verbose) {
      printf("Robot %u could not construct G matrix. Skip update...\n", getID());
    }
    return false;
  }

  // Initialize optimizer
  QuadraticOptimizer optimizer(mProblemPtr);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setAlgorithm(mParams.algorithm);
  optimizer.setTrustRegionTolerance(1e-2); // Force optimizer to make progress
  optimizer.setTrustRegionIterations(1);
  optimizer.setTrustRegionMaxInnerIterations(10);
  optimizer.setTrustRegionInitialRadius(100);

  // Starting solution
  Matrix XInit;
  if (acceleration) {
    XInit = Y;
  } else {
    XInit = X;
  }
  assert(XInit.rows() == relaxation_rank());
  assert(XInit.cols() == (dimension() + 1) * num_poses());

  // Optimize!
  X = optimizer.optimize(XInit);
  assert(X.rows() == relaxation_rank());
  assert(X.cols() == (dimension() + 1) * num_poses());

  // Print optimization statistics
  const auto &result = optimizer.getOptResult();
  if (mParams.verbose) {
    printf("df: %f, gn0: %f, gn1: %f, df/gn0: %f\n",
           result.fInit - result.fOpt,
           result.gradNormInit,
           result.gradNormOpt,
           (result.fInit - result.fOpt) / result.gradNormInit);
  }

  return true;
}

void PGOAgent::resetTeamStatus() {
  mTeamStatus.clear();
  for (unsigned robot = 0; robot < mParams.numRobots; ++robot) {
    mTeamStatus.emplace_back(robot);
  }
}

bool PGOAgent::shouldUpdateLoopClosureWeights() const {
  // No need to update weight if using L2 cost
  if (mParams.robustCostType == RobustCostType::L2) return false;

  return ((mIterationNumber + 1) % mParams.robustOptInnerIters == 0);
}

void PGOAgent::updateLoopClosuresWeights() {
  assert(mState == PGOAgentState::INITIALIZED);

  // Update private loop closures
  for (auto &m: privateLoopClosures) {
    if (m.isKnownInlier) continue;
    Matrix Y1 = X.block(0, m.p1 * (d + 1), r, d);
    Matrix p1 = X.block(0, m.p1 * (d + 1) + d, r, 1);
    Matrix Y2 = X.block(0, m.p2 * (d + 1), r, d);
    Matrix p2 = X.block(0, m.p2 * (d + 1) + d, r, 1);
    double residual = std::sqrt(computeMeasurementError(m, Y1, p1, Y2, p2));
    double weight = mRobustCost.weight(residual);
    m.weight = weight;
    if (mParams.verbose) {
      printf("Agent %u update edge: (%zu, %zu) -> (%zu, %zu), residual = %f, weight = %f \n",
             getID(), m.r1, m.p1, m.r2, m.p2, residual, weight);
    }
  }

  // Update shared loop closures
  // Agent i is only responsible for updating edge weights with agent j, where j > i
  for (auto &m: sharedLoopClosures) {
    if (m.isKnownInlier) continue;
    Matrix Y1, Y2, p1, p2;
    if (m.r1 == getID()) {
      if (m.r2 < getID()) continue;

      Y1 = X.block(0, m.p1 * (d + 1), r, d);
      p1 = X.block(0, m.p1 * (d + 1) + d, r, 1);
      const PoseID nbrPoseID = std::make_pair(m.r2, m.p2);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        printf("Agent %u cannot update edge: (%zu, %zu) -> (%zu, %zu). \n",
               getID(), m.r1, m.p1, m.r2, m.p2);
        continue;
      }
      Matrix X2 = KVpair->second;
      Y2 = X2.block(0, 0, r, d);
      p2 = X2.block(0, d, r, 1);
    } else {
      if (m.r1 < getID()) continue;

      Y2 = X.block(0, m.p2 * (d + 1), r, d);
      p2 = X.block(0, m.p2 * (d + 1) + d, r, 1);
      const PoseID nbrPoseID = std::make_pair(m.r1, m.p1);
      auto KVpair = neighborPoseDict.find(nbrPoseID);
      if (KVpair == neighborPoseDict.end()) {
        printf("Agent %u cannot update edge: (%zu, %zu) -> (%zu, %zu). \n",
               getID(), m.r1, m.p1, m.r2, m.p2);
        continue;
      }
      Matrix X1 = KVpair->second;
      Y1 = X1.block(0, 0, r, d);
      p1 = X1.block(0, d, r, 1);
    }
    double residual = std::sqrt(computeMeasurementError(m, Y1, p1, Y2, p2));
    double weight = mRobustCost.weight(residual);
    m.weight = weight;
    if (mParams.verbose) {
      printf("Agent %u update edge: (%zu, %zu) -> (%zu, %zu), residual = %f, weight = %f \n",
             getID(), m.r1, m.p1, m.r2, m.p2, residual, weight);
    }
  }
  mPublishWeightsRequested = true;
}

double PGOAgent::computeConvergedLoopClosureRatio() {
  // Currently, this function is only meaningful for GNC_TLS
  if (mParams.robustCostType != RobustCostType::GNC_TLS) {
    return 1.0;
  }

  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  for (const auto &m: privateLoopClosures) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  for (const auto &m: sharedLoopClosures) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  double convergedCount = acceptCount + rejectCount;

  if (mParams.verbose) {
    printf(
        "Robot %u :\n "
        "accepted loop closures: %i\n "
        "rejected loop closures: %i\n "
        "undecided loop closures: %i\n",
        getID(),
        (int) acceptCount,
        (int) rejectCount,
        (int) (totalCount - convergedCount));
  }

  return convergedCount / totalCount;
}

bool PGOAgent::isDuplicateMeasurement(const RelativeSEMeasurement &m,
                                      const vector<RelativeSEMeasurement> &measurements) {
  for (const RelativeSEMeasurement &m2: measurements) {
    if (m.r1 == m2.r1 && m.r2 == m2.r2 && m.p1 == m2.p1 && m.p2 == m2.p2) {
      return true;
    }
  }
  return false;
}

}  // namespace DPGO