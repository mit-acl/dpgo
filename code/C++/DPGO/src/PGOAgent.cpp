/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticOptimizer.h>
#include <DPGO/QuadraticProblem.h>

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
    : mID(ID), mCluster(ID), d(params.d), r(params.r), n(1),
      mParams(params), mState(PGOAgentState::WAIT_FOR_DATA),
      mStatus(ID, mState, 0, 0, false, 0),
      mInstanceNumber(0), mIterationNumber(0), mNumPosesReceived(0),
      logger(params.logDirectory) {
  if (mParams.verbose) std::cout << mParams << std::endl;
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
  mCluster = 0;
  X = Xin;
  if (mParams.acceleration) {
    XPrev = X;
    initializeAcceleration();
  }
  if (mParams.verbose)
    std::cout << "WARNING: Agent " << mID
              << " resets trajectory. New trajectory length: " << n
              << std::endl;
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
  for (const auto &mSharedPose : mSharedPoses) {
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
  for (const auto &mSharedPose : mSharedPoses) {
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
    const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures) {
  assert(!isOptimizationRunning());
  assert(mState == PGOAgentState::WAIT_FOR_DATA);
  assert(n == 1);

  if (inputOdometry.empty()) return;

  for (const auto &edge : inputOdometry) {
    addOdometry(edge);
  }
  for (const auto &edge : inputPrivateLoopClosures) {
    addPrivateLoopClosure(edge);
  }
  for (const auto &edge : inputSharedLoopClosures) {
    addSharedLoopClosure(edge);
  }

  mState = PGOAgentState::WAIT_FOR_INITIALIZATION;

  if (mParams.logData) {
    std::vector<RelativeSEMeasurement> measurements = odometry;
    measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());
    measurements.insert(measurements.end(), sharedLoopClosures.begin(), sharedLoopClosures.end());
    logger.logMeasurements(measurements, "measurements.csv");
  }

  if (mID == 0) {
    // The first agent can further initialize its trajectory estimate
    Matrix T = localChordalInitialization();
    X = YLift.value() * T;  // Lift to correct relaxation rank
    mState = PGOAgentState::INITIALIZED;
    if (mParams.acceleration) {
      XPrev = X;
      initializeAcceleration();
    }

    // Save initial trajectory
    if (mParams.logData) {
      logger.logTrajectory(dimension(), num_poses(), T, "trajectory_initial.csv");
    }
  }

  constructQMatrix();
}

void PGOAgent::constructQMatrix() {
  vector<RelativeSEMeasurement> privateMeasurements = odometry;
  privateMeasurements.insert(privateMeasurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());
  vector<RelativeSEMeasurement> sharedMeasurements = sharedLoopClosures;

  Q = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d + 1, d + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d + 1, d + 1);

  for (auto m : sharedMeasurements) {

    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Set aggregate weight matrix
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.kappa;
    }
    Omega(d, d) = m.tau;

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
    mSharedPoses.insert(std::make_pair(mID, factor.p1));
    neighborSharedPoses.insert(std::make_pair(factor.r2, factor.p2));
    neighborAgents.insert(factor.r2);
  } else {
    assert(factor.r2 == mID);
    n = std::max(n, (unsigned) factor.p2 + 1);
    mSharedPoses.insert(std::make_pair(mID, factor.p2));
    neighborSharedPoses.insert(std::make_pair(factor.r1, factor.p1));
    neighborAgents.insert(factor.r1);
  }

  lock_guard<mutex> lock(mMeasurementsMutex);
  sharedLoopClosures.push_back(factor);
}

void PGOAgent::updateNeighborPose(unsigned neighborCluster, unsigned neighborID,
                                  unsigned neighborPose, const Matrix &var) {
  assert(neighborID != mID);
  assert(var.rows() == r);
  assert(var.cols() == d + 1);

  PoseID nID = std::make_pair(neighborID, neighborPose);

  mNumPosesReceived++;

  // Do not store this pose if not needed
  if (neighborSharedPoses.find(nID) == neighborSharedPoses.end()) {
    return;
  }

  // Check if this agent is ready to initialize
  if (mState == PGOAgentState::WAIT_FOR_INITIALIZATION) {
    if (neighborCluster == 0) {
      std::cout << "Agent " << mID << " informed by agent " << neighborID
                << " to initialize! " << std::endl;

      // Require the lifting matrix to initialize
      assert(YLift);

      // Halt optimization
      bool optimizationHalted = false;
      if (isOptimizationRunning()) {
        if (mParams.verbose)
          std::cout << "Agent " << mID << " halt optimization thread..." << std::endl;
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

      // Find the corresponding inter-robot loop closure
      RelativeSEMeasurement m;
      bool found = findSharedLoopClosure(neighborID, neighborPose, m);
      assert(found);

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
      Matrix T = localChordalInitialization();
      Matrix T_frame1_frame2 = Matrix::Identity(d + 1, d + 1);
      Matrix T_world1_frame1 = Matrix::Identity(d + 1, d + 1);
      if (m.r1 == neighborID) {
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

      Matrix T_world1_frame = Matrix::Identity(d + 1, d + 1);
      Matrix T_world2_frame = Matrix::Identity(d + 1, d + 1);
      for (size_t i = 0; i < n; ++i) {
        T_world1_frame.block(0, 0, d, d + 1) =
            T.block(0, i * (d + 1), d, d + 1);
        T_world2_frame = T_world2_world1 * T_world1_frame;
        T.block(0, i * (d + 1), d, d + 1) =
            T_world2_frame.block(0, 0, d, d + 1);
      }

      // Lift back to correct relaxation rank
      X = YLift.value() * T;

      // Mark this agent as initialized
      mCluster = neighborCluster;
      mState = PGOAgentState::INITIALIZED;

      // Initialize auxiliary variables
      if (mParams.acceleration) {
        XPrev = X;
        initializeAcceleration();
      }

      // Log initial trajectory
      if (mParams.logData) {
        logger.logTrajectory(dimension(), num_poses(), T, "trajectory_initial.csv");
      }

      if (optimizationHalted) startOptimizationLoop(rate);
    }
  }

  // Only save poses from neighbors if this agent is initialized
  // and if the sending agent is also initialized
  if (mState == PGOAgentState::INITIALIZED && neighborCluster == 0) {
    lock_guard<mutex> lock(mNeighborPosesMutex);
    neighborPoseDict[nID] = var;
  }
}

void PGOAgent::updateAuxNeighborPose(unsigned neighborCluster, unsigned neighborID,
                                     unsigned neighborPose, const Matrix &var) {
  assert(mParams.acceleration);
  assert(neighborID != mID);
  assert(var.rows() == r);
  assert(var.cols() == d + 1);

  PoseID nID = std::make_pair(neighborID, neighborPose);

  mNumPosesReceived++;

  // Do not store this pose if not needed
  if (neighborSharedPoses.find(nID) == neighborSharedPoses.end()) return;

  // Only save poses from neighbors if this agent is initialized
  // and if the sending agent is also initialized
  if (mState == PGOAgentState::INITIALIZED && neighborCluster == 0) {
    lock_guard<mutex> lock(mNeighborPosesMutex);
    neighborAuxPoseDict[nID] = var;
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

std::vector<unsigned> PGOAgent::getNeighborPublicPoses(
    const unsigned &neighborID) const {
  // Check that neighborID is indeed a neighbor of this agent
  assert(neighborAgents.find(neighborID) != neighborAgents.end());
  std::vector<unsigned> poseIndices;
  for (PoseID pair : neighborSharedPoses) {
    if (pair.first == neighborID) {
      poseIndices.push_back(pair.second);
    }
  }
  return poseIndices;
}

std::vector<unsigned> PGOAgent::getNeighbors() const {
  std::vector<unsigned> v(neighborAgents.size());
  std::copy(neighborAgents.begin(), neighborAgents.end(), v.begin());
  return v;
}

void PGOAgent::reset() {
  // Terminate optimization thread if running
  endOptimizationLoop();

  if (mParams.logData) {
    Matrix T;
    if (getTrajectoryInGlobalFrame(T)) {
      logger.logTrajectory(dimension(), num_poses(), T, "trajectory_optimized.csv");
    }

    // Save solution before rounding
    std::string filename = mParams.logDirectory + "X.txt";
    std::ofstream file(filename);
    file << X ;
  }

  mInstanceNumber++;
  mIterationNumber = 0;
  mNumPosesReceived = 0;

  // Assume that the old lifting matrix can still be used
  mState = PGOAgentState::WAIT_FOR_DATA;
  mStatus = PGOAgentStatus(getID(), getState(), mInstanceNumber, mIterationNumber, false, 0);

  odometry.clear();
  privateLoopClosures.clear();
  sharedLoopClosures.clear();

  neighborPoseDict.clear();
  neighborAuxPoseDict.clear();
  mSharedPoses.clear();
  neighborSharedPoses.clear();
  neighborAgents.clear();
  resetTeamStatus();

  n = 1;
  X = Matrix::Zero(r, d + 1);
  X.block(0, 0, d, d) = Matrix::Identity(d, d);

  mCluster = mID;
}

void PGOAgent::iterate(bool doOptimization) {
  mIterationNumber++;

  // Save early stopped solution
  if (mIterationNumber == 50 && mParams.logData) {
    Matrix T;
    if (getTrajectoryInGlobalFrame(T)) {
      logger.logTrajectory(dimension(), num_poses(), T, "trajectory_early_stop.csv");
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
      // Periodic restart
      if (shouldRestart()) {
        if (mParams.verbose) std::cout << "Restart!" << std::endl;
        X = XPrev;
        updateX(doOptimization, false);
        V = X;
        Y = X;
        gamma = 0;
        alpha = 0;
      }
    } else {
      success = updateX(doOptimization, false);
    }

    // Update status
    if (doOptimization) {
      mStatus.agentID = getID();
      mStatus.state = getState();
      mStatus.instanceNumber = instance_number();
      mStatus.iterationNumber = iteration_number();
      mStatus.optimizationSuccess = success;
      mStatus.relativeChange = sqrt((X - XPrev).squaredNorm() / num_poses());
    }
  }
}

bool PGOAgent::constructGMatrix(SparseMatrix &G, const PoseDict &poseDict) {
  G.setZero();
  assert(G.rows() == relaxation_rank());
  assert(G.cols() == (dimension()+1) * num_poses());

  for (auto m : sharedLoopClosures) {
    // Construct relative SE matrix in homogeneous form
    Matrix T = Matrix::Zero(d + 1, d + 1);
    T.block(0, 0, d, d) = m.R;
    T.block(0, d, d, 1) = m.t;
    T(d, d) = 1;

    // Construct aggregate weight matrix
    Matrix Omega = Matrix::Zero(d + 1, d + 1);
    for (unsigned row = 0; row < d; ++row) {
      Omega(row, row) = m.kappa;
    }
    Omega(d, d) = m.tau;

    if (m.r1 == mID) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      assert(m.r2 != mID);

      // Read neighbor's pose
      const PoseID nID = std::make_pair(m.r2, m.p2);
      auto KVpair = poseDict.find(nID);
      if (KVpair == poseDict.end()) {
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

  return true;
}

void PGOAgent::startOptimizationLoop(double freq) {
  // Asynchronous updates currently restricted to non-accelerated updates
  assert(!mParams.acceleration);

  if (isOptimizationRunning()) {
    if (mParams.verbose)
      std::cout << "WARNING: optimization thread already running! Skip..." << std::endl;
    return;
  }

  rate = freq;

  mOptimizationThread = new thread(&PGOAgent::runOptimizationLoop, this);
}

void PGOAgent::runOptimizationLoop() {
  if (mParams.verbose)
    std::cout << "Agent " << mID << " optimization thread running at " << rate
              << " Hz." << std::endl;

  // Create exponential distribution with the desired rate
  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::exponential_distribution<double> ExponentialDistribution(rate);

  while (true) {
    double sleepUs =
        1e6 * ExponentialDistribution(rng);  // sleeping time in microsecond

    usleep(sleepUs);

    iterate(true);

    // Check if finish requested
    if (mFinishRequested) {
      break;
    }
  }
}

void PGOAgent::endOptimizationLoop() {
  if (!isOptimizationRunning()) return;

  mFinishRequested = true;

  // wait for thread to finish
  mOptimizationThread->join();

  delete mOptimizationThread;

  mOptimizationThread = nullptr;

  mFinishRequested = false;  // reset request flag

  if (mParams.verbose)
    std::cout << "Agent " << mID << " optimization thread exited. " << std::endl;
}

bool PGOAgent::isOptimizationRunning() {
  return mOptimizationThread != nullptr;
}

bool PGOAgent::findSharedLoopClosure(unsigned neighborID, unsigned neighborPose,
                                     RelativeSEMeasurement &mOut) {
  for (const auto &m : sharedLoopClosures) {
    if ((m.r1 == neighborID && m.p1 == neighborPose) ||
        (m.r2 == neighborID && m.p2 == neighborPose)) {
      mOut = m;
      return true;
    }
  }

  return false;
}

Matrix PGOAgent::localChordalInitialization() {
  std::vector<RelativeSEMeasurement> measurements = odometry;
  measurements.insert(measurements.end(), privateLoopClosures.begin(),
                      privateLoopClosures.end());

  return chordalInitialization(dimension(), num_poses(), measurements);
}

Matrix PGOAgent::localPoseGraphOptimization() {

  // Compute initialization
  Matrix Tinit = localChordalInitialization();
  assert(Tinit.rows() == d);
  assert(Tinit.cols() == (d + 1) * n);

  // Construct data matrix
  SparseMatrix G(d, (d + 1) * n);  // linear terms should be zero

  // Form optimization problem
  QuadraticProblem problem(n, d, d, Q, G);

  // Initialize optimizer object
  QuadraticOptimizer optimizer(&problem);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setTrustRegionIterations(10);
  optimizer.setTrustRegionTolerance(1e-2);

  // Optimize
  Matrix Topt = optimizer.optimize(Tinit);
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
    std::cout << "Reached maximum iterations." << std::endl;
    return true;
  }

  // terminate if all agents satisfy relative change condition
  bool relative_change_reached = true;
  for (size_t robot = 0; robot < mParams.numRobots; ++robot) {
    PGOAgentStatus robotStatus = mTeamStatus[robot];
    if (robotStatus.agentID != robot ||
        !robotStatus.optimizationSuccess ||
        robotStatus.relativeChange > mParams.relChangeTol) {
      relative_change_reached = false;
      break;
    }
  }
  if (relative_change_reached) {
    std::cout << "Reached relative change stopping condition." << std::endl;
    return true;
  }

  return false;
}

bool PGOAgent::shouldRestart() {
  assert(mParams.acceleration);
  return (mIterationNumber % mParams.restartInterval == 0);
}

void PGOAgent::initializeAcceleration() {
  assert(mParams.acceleration);
  if (mState == PGOAgentState::INITIALIZED) {
    X = XPrev;
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

  if (mParams.verbose) std::cout << "Agent " << mID << " optimize..." << std::endl;

  if (acceleration)
    assert(mParams.acceleration);

  assert(mState == PGOAgentState::INITIALIZED);

  // Construct data matrices
  SparseMatrix G(r, (d + 1) * n);
  if (acceleration) {
    if (!constructGMatrix(G, neighborAuxPoseDict)) {
      if (mParams.verbose)
        std::cout << "Could not create cost matrices from auxiliary variables! Skip optimization..."
                  << std::endl;
      return false;
    }
  } else {
    if (!constructGMatrix(G, neighborPoseDict)) {
      if (mParams.verbose)
        std::cout << "Could not create cost matrices! Skip optimization..."
                  << std::endl;
      return false;
    }
  }

  // Initialize optimization problem
  QuadraticProblem problem(n, d, r, Q, G);

  // Initialize optimizer
  QuadraticOptimizer optimizer(&problem);
  optimizer.setVerbose(mParams.verbose);
  optimizer.setAlgorithm(mParams.algorithm);
  optimizer.setTrustRegionTolerance(1e-6); // Force optimizer to make progress
  optimizer.setTrustRegionIterations(1);

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

  return true;
}

void PGOAgent::resetTeamStatus() {
  mTeamStatus.clear();
  for (unsigned robot = 0; robot < mParams.numRobots; ++robot) {
    mTeamStatus.push_back(PGOAgentStatus(robot));
  }
}

}  // namespace DPGO
