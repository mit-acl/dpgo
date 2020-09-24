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
Defines the possible states of a PGOAgent
Each state can only transition to the state below
*/
enum PGOAgentState {

  WAIT_FOR_LIFTING_MATRIX,  // waiting to receive shared lifting matrix

  WAIT_FOR_DATA,  // waiting to receive pose graph

  WAIT_FOR_INITIALIZATION,  // waiting to initialize trajectory estimate

  INITIALIZED,  // trajectory initialized and ready to update

};

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

  /**
  Get internal solution
  */
  Matrix getX() {
    assert(mState != PGOAgentState::WAIT_FOR_LIFTING_MATRIX && mState != PGOAgentState::WAIT_FOR_DATA);
    lock_guard<mutex> lock(mPosesMutex);
    return X;
  }

  /**
  Get the ith component of the current solution
  */
  bool getXComponent(const unsigned index, Matrix& Mout) {
    assert(mState != PGOAgentState::WAIT_FOR_LIFTING_MATRIX && mState != PGOAgentState::WAIT_FOR_DATA);
    lock_guard<mutex> lock(mPosesMutex);
    if (index >= num_poses()) return false;
    Mout = X.block(0, index * (d + 1), r, d + 1);
    return true;
  }

  /**
  Initialize the local pose graph from the input factors
  */
  void setPoseGraph(
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
  Reset this agent to have empty pose graph
  */
  void reset();

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
  Get set of neighboring agents
  */
  inline std::set<unsigned> getNeighbors() const { return neighborAgents; }

  /** 
  * Get vector of pose indices needed from the neighbor agent
  */ 
  std::vector<unsigned> getNeighborPublicPoses(const unsigned& neighborID) const;  

  /**
  Get random neighbor.
  If the agent has no neighbor, this method will return false.
  */
  bool getRandomNeighbor(unsigned& neighborID) const;

  /**
  Return trajectory estimate of this robot in local frame, with its first pose
  set to identity
  */
  Matrix getTrajectoryInLocalFrame();

  /**
  Return trajectory estimate of this robot in global frame, with the first pose
  of robot 0 set to identity
  */
  Matrix getTrajectoryInGlobalFrame(const Matrix& globalAnchor);

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
  Get lifting matrix
  */
  inline Matrix getLiftingMatrix() const {
    assert(mID == 0);
    assert(mState != PGOAgentState::WAIT_FOR_LIFTING_MATRIX);
    return YLift;
  }

  /**
  Set the lifting matrix
  */
  void setLiftingMatrix(const Matrix& Y) {
    assert(mState == PGOAgentState::WAIT_FOR_LIFTING_MATRIX);
    assert(Y.rows() == r);
    assert(Y.cols() == d);
    YLift = Y;
    mState = PGOAgentState::WAIT_FOR_DATA;
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

  // Lifting matrix shared by all agents
  Matrix YLift;

  // Current state of this agent
  PGOAgentState mState;

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

  // Store the set of neighboring agents
  set<unsigned> neighborAgents;

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