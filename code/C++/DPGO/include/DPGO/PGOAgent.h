/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef PGOAGENT_H
#define PGOAGENT_H

#include <DPGO/DPGO_types.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <Eigen/Dense>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "Manifolds/Element.h"
#include "Manifolds/Manifold.h"

using std::lock_guard;
using std::mutex;
using std::set;
using std::thread;
using std::vector;

/*Define the namespace*/
namespace DPGO {

/**
This struct contains parameters for PGOAgent
*/
struct PGOAgentParameters {
  // Problem dimension
  unsigned d;

  // Relaxed rank in Riemanian optimization
  unsigned r;

  // Riemannian optimization algorithm
  ROPTALG algorithm;

  // Verbosility flag
  bool verbose;

  // Default constructor
  PGOAgentParameters(unsigned dIn, unsigned rIn,
                     ROPTALG algorithmIn = ROPTALG::RTR, bool v = true)
      : d(dIn), r(rIn), algorithm(algorithmIn), verbose(v) {}
};

class PGOAgent {
 public:
  /** Constructor that creates an empty pose graph
   */
  PGOAgent(unsigned ID, const PGOAgentParameters& params);

  ~PGOAgent();

  /** Helper function to reset the internal solution
      In deployment, probably should not use this
   */
  void setX(const Matrix& Xin) {
    lock_guard<mutex> lock(mPosesMutex);
    X = Xin;
    n = X.cols() / (d + 1);
    assert(X.cols() == n * (d + 1));
    assert(X.rows() == r);
    if (verbose)
      std::cout << "WARNING: Agent " << mID
                << " resets trajectory. New trajectory length: " << n
                << std::endl;
  }

  /** Helper function to get current solution
      In deployment, probably should not use this
   */
  Matrix getX() {
    lock_guard<mutex> lock(mPosesMutex);
    return X;
  }

  /**
  Get the ith component of the current solution
  */
  Matrix getXComponent(unsigned i) {
    X = getX();
    return X.block(0, i * (d + 1), r, d + 1);
  }

  /**
  Initialize the local pose graph from the input factors
  */
  void initialize(
      const std::vector<RelativeSEMeasurement>& inputOdometry,
      const std::vector<RelativeSEMeasurement>& inputPrivateLoopClosures,
      const std::vector<RelativeSEMeasurement>& inputSharedLoopClosures);

  /**
  Store the pose of a neighboring robot who shares loop closure with this robot
  */
  void updateNeighborPose(unsigned neighborCluster, unsigned neighborID,
                          unsigned neighborPose, const Matrix& var);

  /**
  Optimize pose graph by a single iteration.
  This process use both private and shared factors (communication required for
  the latter)
  */
  ROPTResult optimize();

  /**
  Return the cluster that this robot belongs to
  */
  inline unsigned getCluster() const { return mCluster; }

  /**
  Return ID of this robot
  */
  inline unsigned getID() const { return mID; }

  /**
  Return number of poses of this robot
  */
  inline unsigned num_poses() const { return n; }

  /**
  Get dimension
  */
  inline unsigned dimension() const { return d; }

  /**
  Get relaxation rank
  */
  inline unsigned relaxation_rank() const { return r; }

  /**
  Return trajectory estimate of this robot in local frame, with its first pose
  set to identity
  */
  Matrix getTrajectoryInLocalFrame();

  /**
  Return trajectory estimate of this robot in global frame, with the first pose
  of robot 0 set to identity
  */
  Matrix getTrajectoryInGlobalFrame();

  /**
  Return a map of shared poses of this robot, that need to be sent to others
  */
  PoseDict getSharedPoses();

  /**
  Initiate a new thread that runs runOptimizationLoop()
  */
  void startOptimizationLoop(double freq);

  /**
  Request to terminate optimization loop, if running
  This function also waits until the optimization loop is finished
  */
  void endOptimizationLoop();

  /**
  Check if the optimization thread is running
  */
  bool isOptimizationRunning();

  /**
  Set maximum stepsize during Riemannian optimization (only used by RGD)
  */
  void setStepsize(double s) { stepsize = s; }

  /**
  Set the global anchor, which is used during rounding to put the solution in a
  global reference frame
  */
  void setGlobalAnchor(const Matrix& anchor) {
    assert(anchor.rows() == r);
    assert(anchor.cols() == d + 1);
    globalAnchor = anchor;
  }

 private:
  // The unique ID associated to this robot
  unsigned mID;

  // The cluster that this robot belongs to
  unsigned mCluster;

  // Dimension
  unsigned d;

  // Relaxed rank in Riemanian optimization problem
  unsigned r;

  // Number of poses
  unsigned n;

  // Verbose flag
  bool verbose;

  // Rate in Hz of the optimization loop
  double rate;

  // whether there is request to terminate optimization thread
  bool mFinishRequested = false;

  // Optimization algorithm
  ROPTALG algorithm;

  // step size (only used in RGD)
  double stepsize;

  // Solution before rounding
  Matrix X;

  // used during rounding to put the current solution to a global reference
  // frame
  Matrix globalAnchor;

  // Lifting matrix shared by all agents
  Matrix YLift;

  // Store odometric measurement of this robot
  vector<RelativeSEMeasurement> odometry;

  // Store private loop closures of this robot
  vector<RelativeSEMeasurement> privateLoopClosures;

  // This dictionary stores poses owned by other robots that is connected to
  // this robot by loop closure
  PoseDict neighborPoseDict;

  // Store the set of public poses that need to be sent to other robots
  set<PoseID> mSharedPoses;

  // Store the set of public poses needed from other robots
  set<PoseID> neighborSharedPoses;

  // This dictionary stores shared loop closure measurements
  vector<RelativeSEMeasurement> sharedLoopClosures;

  // Implement locking to synchronize read & write of trajectory estimate
  mutex mPosesMutex;

  // Implement locking to synchronize read & write of shared poses from
  // neighbors
  mutex mNeighborPosesMutex;

  // Implement locking on measurements
  mutex mMeasurementsMutex;

  // Thread that runs optimization loop
  thread* mOptimizationThread = nullptr;

  /**
  Add an odometric measurement of this robot.
  This function automatically initialize the new pose, by propagating odometry
  */
  void addOdometry(const RelativeSEMeasurement& factor);

  /**
  Add a private loop closure of this robot
  (Warning: this function does not check for duplicate loop closures!)
  */
  void addPrivateLoopClosure(const RelativeSEMeasurement& factor);

  /**
  Add a shared loop closure between this robot and another
  (Warning: this function does not check for duplicate loop closures!)
  */
  void addSharedLoopClosure(const RelativeSEMeasurement& factor);

  /** Compute the cost matrices that define the local PGO problem
      f(X) = 0.5<Q, XtX> + <X, G>
  */
  bool constructCostMatrices(
      const vector<RelativeSEMeasurement>& privateMeasurements,
      const vector<RelativeSEMeasurement>& sharedMeasurements, SparseMatrix* Q,
      SparseMatrix* G);

  /**
  Optimize pose graph by calling optimize().
  This function will run in a separate thread.
  */
  void runOptimizationLoop();

  /**
  Find a shared loop closure based on neighboring robot's ID and pose
  */
  bool findSharedLoopClosure(unsigned neighborID, unsigned neighborPose,
                             RelativeSEMeasurement& mOut);

  /**
  local Chordal initialization
  */
  Matrix computeInitialEstimate();
};

}  // namespace DPGO

#endif