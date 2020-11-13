/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef PGOAGENT_H
#define PGOAGENT_H

#include <DPGO/DPGO_types.h>
#include <DPGO/PGOLogger.h>
#include <DPGO/RelativeSEMeasurement.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>

#include <Eigen/Dense>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <utility>
#include <vector>
#include <optional>

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

  // Relaxed rank in Riemannian optimization
  unsigned r;

  // Total number of robots
  unsigned numRobots;

  // Riemannian optimization algorithm
  ROPTALG algorithm;

  // Stopping conditions
  unsigned maxNumIters;  // Maximum number of global iterations
  double relChangeTol;  // Tolerance on relative change
  double funcDecreaseTol; // Tolerance on function decrease

  // Verbose flag
  bool verbose;

  // Flag to enable data logging
  bool logData;

  // Directory to log data
  std::string logDirectory;

  // Default constructor
  PGOAgentParameters(unsigned dIn, unsigned rIn, unsigned numRobotsIn = 1,
                     ROPTALG algorithmIn = ROPTALG::RTR,
                     unsigned maxIters = 500, double changeTol = 1e-3, double funcDecTol = 1e-5,
                     bool v = false, bool log = false, std::string logDir = "")
      : d(dIn), r(rIn), numRobots(numRobotsIn),
        algorithm(algorithmIn),
        maxNumIters(maxIters), relChangeTol(changeTol), funcDecreaseTol(funcDecTol),
        verbose(v), logData(log), logDirectory(std::move(logDir)) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const PGOAgentParameters &params) {
    os << "PGOAgent parameters: " << std::endl;
    os << "dimension: " << params.d << std::endl;
    os << "relaxation rank: " << params.r << std::endl;
    os << "number of robots: " << params.numRobots << std::endl;
    os << "algorithm: " << params.algorithm << std::endl;
    os << "max iterations: " << params.maxNumIters << std::endl;
    os << "relative change tol: " << params.relChangeTol << std::endl;
    os << "function decrease tol: " << params.funcDecreaseTol << std::endl;
    os << "verbose: " << params.verbose << std::endl;
    os << "log data: " << params.logData << std::endl;
    os << "log directory: " << params.logDirectory << std::endl;
    return os;
  }
};

class PGOAgent {
 public:
  /** Constructor that creates an empty pose graph
   */
  PGOAgent(unsigned ID, const PGOAgentParameters &params);

  ~PGOAgent();

  /**
  Initialize the local pose graph from the input factors
  */
  void setPoseGraph(
      const std::vector<RelativeSEMeasurement> &inputOdometry,
      const std::vector<RelativeSEMeasurement> &inputPrivateLoopClosures,
      const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures);

  /**
   * @brief Iterate and performs necessary bookkeeping
   */
  void iterate();

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
  Return ID of this robot
  */
  inline unsigned getID() const { return mID; }

  /**
  Return the cluster that this robot belongs to
  */
  inline unsigned getCluster() const { return mCluster; }

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
   * @brief Get current instance number
   */
  inline unsigned instance_number() const { return mInstanceNumber; }

  /**
   * @brief Get current global iteration number
   */
  inline unsigned iteration_number() const { return mIterationNumber; }

  /**
   * @brief get the current state of this agent
   * @return PGOAgentState struct
   */
  inline PGOAgentState getState() const { return mState; }

  /**
   * Get vector of pose indices needed from the neighbor agent
   */
  std::vector<unsigned> getNeighborPublicPoses(
      const unsigned &neighborID) const;

  /**
  Get vector of neighbor robot IDs.
  */
  std::vector<unsigned> getNeighbors() const;

  /**
  Return trajectory estimate of this robot in local frame, with its first pose
  set to identity
  */
  bool getTrajectoryInLocalFrame(Matrix &Trajectory);

  /**
  Return trajectory estimate of this robot in global frame, with the first pose
  of robot 0 set to identity
  */
  bool getTrajectoryInGlobalFrame(Matrix &Trajectory);

  /**
  Return a map of shared poses of this robot, that need to be sent to others
  */
  PoseDict getSharedPoses();

  /** Helper function to reset the internal solution
    In deployment, probably should not use this
 */
  void setX(const Matrix &Xin);

  /**
  Get internal solution
  */
  bool getX(Matrix &Mout);

  /**
  Get the ith component of the current solution
  */
  bool getXComponent(unsigned index, Matrix &Mout);

  /**
   * @brief determine if the termination condition is satisfied
   * @return boolean
   */
  bool shouldTerminate();

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
  Get lifting matrix
  */
  bool getLiftingMatrix(Matrix &M) const;

  /**
  Set the lifting matrix
  */
  void setLiftingMatrix(const Matrix &Y);

  /**
  Set the global anchor
  */
  void setGlobalAnchor(const Matrix &M);

  /**
 * @brief Update local copy of a neighbor agent's pose
 * @param neighborCluster the cluster the neighbor agent belongs to
 * @param neighborID the ID of the neighbor agent
 * @param neighborPose local index of the received neighbor pose
 * @param var Actual value of the received pose
 */
  void updateNeighborPose(unsigned neighborCluster, unsigned neighborID,
                          unsigned neighborPose, const Matrix &var);

 protected:
  // The unique ID associated to this robot
  unsigned mID;

  // The cluster that this robot belongs to
  unsigned mCluster;

  // Dimension
  unsigned d;

  // Relaxed rank in Riemannian optimization problem
  unsigned r;

  // Number of poses
  unsigned n;

  // Current state of this agent
  PGOAgentState mState;

  // Parameter settings
  const PGOAgentParameters mParams;

  // Rate in Hz of the optimization loop (only used in asynchronous mode)
  double rate;

  // Current PGO instance
  unsigned mInstanceNumber;

  // Current global iteration counter (this is only meaningful in synchronous mode)
  unsigned mIterationNumber;

  // Total number of neighbor poses received
  unsigned mNumPosesReceived;

  // Logging
  PGOLogger logger;

  // Data structures needed to check termination condition
  std::vector<double> relativeChanges;
  std::vector<double> funcDecreases;

  // whether there is request to terminate optimization thread
  bool mFinishRequested = false;

  // Solution before rounding
  Matrix X;

  // Lifting matrix shared by all agents
  std::optional<Matrix> YLift;

  // Anchor matrix shared by all agents
  std::optional<Matrix> globalAnchor;

  // Store odometry measurement of this robot
  vector<RelativeSEMeasurement> odometry;

  // Store private loop closures of this robot
  vector<RelativeSEMeasurement> privateLoopClosures;

  // Store shared loop closure measurements
  vector<RelativeSEMeasurement> sharedLoopClosures;

  // This dictionary stores poses owned by other robots that is connected to
  // this robot by loop closure
  PoseDict neighborPoseDict;

  // Store the set of public poses that need to be sent to other robots
  set<PoseID> mSharedPoses;

  // Store the set of public poses needed from other robots
  set<PoseID> neighborSharedPoses;

  // Store the set of neighboring agents
  set<unsigned> neighborAgents;

  // Implement locking to synchronize read & write of trajectory estimate
  mutex mPosesMutex;

  // Implement locking to synchronize read & write of shared poses from
  // neighbors
  mutex mNeighborPosesMutex;

  // Implement locking on measurements
  mutex mMeasurementsMutex;

  // Thread that runs optimization loop
  thread *mOptimizationThread = nullptr;

  /**
  Add an odometric measurement of this robot.
  This function automatically initialize the new pose, by propagating odometry
  */
  void addOdometry(const RelativeSEMeasurement &factor);

  /**
  Add a private loop closure of this robot
  (Warning: this function does not check for duplicate loop closures!)
  */
  void addPrivateLoopClosure(const RelativeSEMeasurement &factor);

  /**
  Add a shared loop closure between this robot and another
  (Warning: this function does not check for duplicate loop closures!)
  */
  void addSharedLoopClosure(const RelativeSEMeasurement &factor);

  /**
   * @brief Construct the cost matrices that define the local PGO problem
      f(X) = 0.5<Q, XtX> + <X, G>
   * @param Q: the quadratic data matrix that will be modified in place
   * @param G: the linear data matrix that will be modified in place
   * @return true if the data matrices are computed successfully
   */
  bool constructCostMatrices(SparseMatrix &Q, SparseMatrix &G);

  /**
  Optimize pose graph by calling optimize().
  This function will run in a separate thread.
  */
  void runOptimizationLoop();

  /**
   * @brief Find a shared loop closure based on neighboring robot's ID and pose
   * @param neighborID
   * @param neighborPose
   * @param mOut
   * @return
   */
  bool findSharedLoopClosure(unsigned neighborID, unsigned neighborPose,
                             RelativeSEMeasurement &mOut);

  /**
  Local chordal initialization
  */
  Matrix localChordalInitialization();

  /**
  Local pose graph optimization
  */
  Matrix localPoseGraphOptimization();
};

}  // namespace DPGO

#endif