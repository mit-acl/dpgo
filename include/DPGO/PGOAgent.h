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
#include <DPGO/DPGO_robust.h>
#include <DPGO/QuadraticProblem.h>
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
#include <stdexcept>
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

  // Riemannian optimization algorithm used when solving local subproblem
  ROPTALG algorithm;

  // Cross-robot initialization
  bool multirobot_initialization;

  // Use Nesterov acceleration
  bool acceleration;

  // Interval for fixed (periodic) restart
  unsigned restartInterval;

  // Robust cost function
  RobustCostType robustCostType;

  // Parameter settings over robust cost functions
  RobustCostParameters robustCostParams;

  // Warm start iterate during robust optimization
  bool robustOptWarmStart;

  // Number of inner iterations to apply before updating measurement weights during robust optimization
  unsigned robustOptInnerIters;

  // Minimum ratio of converged weights before terminating robust optimization
  double robustOptMinConvergenceRatio;

  // Maximum number of global iterations
  unsigned maxNumIters;

  // Tolerance on relative change
  double relChangeTol;

  // Verbose flag
  bool verbose;

  // Flag to enable data logging
  bool logData;

  // Directory to log data
  std::string logDirectory;

  // Default constructor
  PGOAgentParameters(unsigned dIn,
                     unsigned rIn,
                     unsigned numRobotsIn = 1,
                     ROPTALG algorithmIn = ROPTALG::RTR,
                     bool accel = false,
                     unsigned restartInt = 30,
                     RobustCostType costType = RobustCostType::L2,
                     RobustCostParameters costParams = RobustCostParameters(),
                     bool robust_opt_warm_start = true,
                     unsigned robust_opt_inner_iters = 30,
                     double robust_opt_min_convergence_ratio = 0.8,
                     unsigned maxIters = 500,
                     double changeTol = 5e-3,
                     bool v = false,
                     bool log = false,
                     std::string logDir = "")
      : d(dIn), r(rIn), numRobots(numRobotsIn),
        algorithm(algorithmIn), multirobot_initialization(true),
        acceleration(accel), restartInterval(restartInt),
        robustCostType(costType), robustCostParams(costParams),
        robustOptWarmStart(robust_opt_warm_start),
        robustOptInnerIters(robust_opt_inner_iters),
        robustOptMinConvergenceRatio(robust_opt_min_convergence_ratio),
        maxNumIters(maxIters), relChangeTol(changeTol),
        verbose(v), logData(log), logDirectory(std::move(logDir)) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const PGOAgentParameters &params) {
    os << "PGOAgent parameters: " << std::endl;
    os << "Dimension: " << params.d << std::endl;
    os << "Relaxation rank: " << params.r << std::endl;
    os << "Number of robots: " << params.numRobots << std::endl;
    os << "Use multi-robot initialization: " << params.multirobot_initialization << std::endl;
    os << "Use Nesterov acceleration: " << params.acceleration << std::endl;
    os << "Fixed restart interval: " << params.restartInterval << std::endl;
    os << "Robust cost function: " << RobustCostNames[params.robustCostType] << std::endl;
    os << "Robust optimization warm start: " << params.robustOptWarmStart << std::endl;
    os << "Robust optimization inner iterations: " << params.robustOptInnerIters << std::endl;
    os << "Robust optimization weight convergence min ratio: " << params.robustOptMinConvergenceRatio << std::endl;
    os << "Local optimization algorithm: " << params.algorithm << std::endl;
    os << "Max iterations: " << params.maxNumIters << std::endl;
    os << "Relative change tol: " << params.relChangeTol << std::endl;
    os << "Verbose: " << params.verbose << std::endl;
    os << "Log data: " << params.logData << std::endl;
    os << "Log directory: " << params.logDirectory << std::endl;
    os << params.robustCostParams << std::endl;
    return os;
  }
};

// Status of an agent to be shared with its peers
struct PGOAgentStatus {
  // Unique ID of this agent
  unsigned agentID;

  // Current state of this agent
  PGOAgentState state;

  // Current problem instance number
  unsigned instanceNumber;

  // Current global iteration number
  unsigned iterationNumber;

  // True if the agent passes its local termination condition
  bool readyToTerminate;

  // The relative change of the agent's estimate
  double relativeChange;

  // Constructor
  explicit PGOAgentStatus(unsigned id,
                          PGOAgentState s = PGOAgentState::WAIT_FOR_DATA,
                          unsigned instance = 0,
                          unsigned iteration = 0,
                          bool ready_to_terminate = false,
                          double relative_change = 0)
      : agentID(id),
        state(s),
        instanceNumber(instance),
        iterationNumber(iteration),
        readyToTerminate(ready_to_terminate),
        relativeChange(relative_change) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const PGOAgentStatus &status) {
    os << "PGOAgent status: " << std::endl;
    os << "ID: " << status.agentID << std::endl;
    os << "State: " << status.state << std::endl;
    os << "Instance number: " << status.instanceNumber << std::endl;
    os << "Iteration number: " << status.iterationNumber << std::endl;
    os << "Ready to terminate: " << status.readyToTerminate << std::endl;
    os << "Relative change: " << status.relativeChange << std::endl;
    return os;
  }
} __attribute__((aligned(32)));

class PGOAgent {
 public:
  /** Constructor that creates an empty pose graph
   */
  PGOAgent(unsigned ID, const PGOAgentParameters &params);

  ~PGOAgent();

  /**
   * @brief Set the local pose graph of this robot, optionally with an initial trajectory estimate
   * @param inputOdometry : odometry edges of this robot
   * @param inputPrivateLoopClosures : internal loop closures of this robot
   * @param inputSharedLoopClosures share : loop closures with other robots
   * @param TInit : optional trajectory estimate [R1 t1 ... Rn tn] in an arbitrary frame. If the  matrix is empty or
   * if its dimension does not match the expected dimension, the value will be discarded and internal initialization
   * will be used instead.
   */
  void setPoseGraph(
      const std::vector<RelativeSEMeasurement> &inputOdometry,
      const std::vector<RelativeSEMeasurement> &inputPrivateLoopClosures,
      const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures,
      const Matrix &TInit = Matrix());

  /**
   * @brief perform a single iteration
   * @param doOptimization: if true, this robot is selected to perform local optimization at this iteration
   */
  void iterate(bool doOptimization = true);

  /**
  Reset this agent to have empty pose graph
  */
  virtual void reset();

  /**
   * @brief Reset variables used in Nesterov acceleration
   */
  void initializeAcceleration();

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
   * @brief Get current instance number
   */
  inline unsigned instance_number() const { return mInstanceNumber; }

  /**
   * @brief Get current global iteration number
   */
  inline unsigned iteration_number() const { return mIterationNumber; }

  /**
   * @brief get the current status of this agent
   * @return
   */
  inline PGOAgentStatus getStatus() {
    mStatus.agentID = getID();
    mStatus.state = mState;
    mStatus.instanceNumber = instance_number();
    mStatus.iterationNumber = iteration_number();
    return mStatus;
  }

  /**
   * @brief get current state of a neighbor
   * @param neighborID
   * @return
   */
  inline PGOAgentStatus getNeighborStatus(unsigned neighborID) const {
    return mTeamStatus[neighborID];
  }

  /**
   * @brief Set the status of a neighbor
   * @param status
   */
  inline void setNeighborStatus(const PGOAgentStatus &status) {
    mTeamStatus[status.agentID] = status;
  }

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
   * @brief Return a single pose in the global frame
   * @param poseID
   * @param T
   * @return
   */
  bool getPoseInGlobalFrame(unsigned poseID, Matrix &T);

  /**
   * @brief Get the pose of a neighbor in global frame
   * @param neighborID
   * @param poseID
   * @param T
   * @return
   */
  bool getNeighborPoseInGlobalFrame(unsigned neighborID, unsigned poseID, Matrix &T);

  /**
   * @brief Get a single public pose of this robot.
   * Note that currently, this method does not check that the requested pose is a public pose
   * @param index: index of the requested pose
   * @param Mout: actual value of the pose
   * @return true if the requested pose exists
   */
  bool getSharedPose(unsigned index, Matrix &Mout);

  /**
   * @brief Get auxiliary variable associated with a single public pose
   * @param index
   * @param Mout
   * @return true if the requested pose exists
   */
  bool getAuxSharedPose(unsigned index, Matrix &Mout);

  /**
   * @brief Get a map of all public poses of this robot
   * @param map: PoseDict object whose content will be filled
   * @return true if the agent is initialized
   */
  bool getSharedPoseDict(PoseDict &map);

  /**
   * @brief Get a map of all auxiliary variables associated with public poses of this robot
   * @param map
   * @return true if agent is initialized
   */
  bool getAuxSharedPoseDict(PoseDict &map);

  /**
   * @brief Helper function to reset internal solution. Currently only for debugging.
   * @param Xin
   */
  void setX(const Matrix &Xin);

  /**
   * @brief Helper function to get internal solution. Note that this method disregards whether the agent is initialized.
   * @param Mout
   * @return
   */
  bool getX(Matrix &Mout);

  /**
   * @brief determine if the termination condition is satisfied
   * @return boolean
   */
  bool shouldTerminate();

  /**
   * @brief Check restart condition
   * @return boolean
   */
  bool shouldRestart() const;

  /**
   * @brief Restart Nesterov acceleration sequence
   * @param doOptimization true if perform optimization after restart
   */
  void restartNesterovAcceleration(bool doOptimization);

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
  void setLiftingMatrix(const Matrix &M);

  /**
  Set the global anchor
  */
  void setGlobalAnchor(const Matrix &M);

  /**
   * @brief
   * @param neighborID
   * @param neighborPose
   * @param var
   * @return
   */
  Matrix computeNeighborTransform(const PoseID &nID, const Matrix &var);

  /**
   * @brief Compute a robust relative transform estimate between this robot and neighbor robot, using a two-stage method
   * which first perform robust single rotation averaging, and then performs translation averaging on the inlier set.
   * @param neighborID
   * @param poseDict
   * @return
   */
  Matrix computeRobustNeighborTransformTwoStage(unsigned neighborID, const PoseDict &poseDict);

  /**
   * @brief Compute a robust relative transform estimate between this robot and neighbor robot, by solving a robust single
   * pose averaging problem using GNC.
   * @param neighborID
   * @param poseDict
   * @return
   */
  Matrix computeRobustNeighborTransform(unsigned neighborID, const PoseDict &poseDict);
  /**
   * @brief Initialize this robot's trajectory estimate in the global frame, using a list of public poses from a neighbor robot
   * @param neighborID
   * @param poseIDs
   * @param vars
   */
  void initializeInGlobalFrame(unsigned neighborID, const PoseDict &poseDict);

  /**
 * @brief Update local copy of a neighbor agent's pose
 * @param neighborID the ID of the neighbor agent
 */
  void updateNeighborPoses(unsigned neighborID, const PoseDict &poseDict);

  /**
   * @brief Update local copy of a neighbor's auxiliary pose
   * @param neighborID
   */
  void updateAuxNeighborPoses(unsigned neighborID, const PoseDict &poseDict);

  /**
   * @brief Perform local PGO using the standard L2 (least-squares) cost function
   * @return trajectory estimate in matrix form T = [R1 t1 ... Rn tn] in an arbitrary frame
   */
  Matrix localPoseGraphOptimization();

 protected:
  // The unique ID associated to this robot
  unsigned mID;

  // Dimension
  unsigned d;

  // Relaxed rank in Riemannian optimization problem
  unsigned r;

  // Number of poses
  unsigned n;

  // Parameter settings
  const PGOAgentParameters mParams;

  // Current state of this agent
  PGOAgentState mState;

  // Current status of this agent (to be shared with others)
  PGOAgentStatus mStatus;

  // Robust cost function
  RobustCost mRobustCost;

  // Pointer to optimization problem
  QuadraticProblem *mProblemPtr;

  // Rate in Hz of the optimization loop (only used in asynchronous mode)
  double mRate{};

  // Current PGO instance
  unsigned mInstanceNumber;

  // Current global iteration counter (this is only meaningful in synchronous mode)
  unsigned mIterationNumber;

  // Total number of neighbor poses received
  unsigned mNumPosesReceived;

  // Logging
  PGOLogger mLogger;

  // Store status of peer agents
  std::vector<PGOAgentStatus> mTeamStatus;

  // Request to perform single local optimization step
  bool mOptimizationRequested = false;

  // Request to publish public poses
  bool mPublishPublicPosesRequested = false;

  // Request to publish measurement weights
  bool mPublishWeightsRequested = false;

  // Request to terminate optimization thread
  bool mEndLoopRequested = false;

  // Solution before rounding
  Matrix X;

  // Initial iterate
  std::optional<Matrix> XInit;

  // Initial solution TInit = [R1 t1 ... Rn tn] in an arbitrary coordinate frame
  std::optional<Matrix> TLocalInit;

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
  set<PoseID> localSharedPoseIDs;

  // Store the set of public poses needed from other robots
  set<PoseID> neighborSharedPoseIDs;

  // Store the set of neighboring agents
  set<unsigned> neighborRobotIDs;

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
  Add an odometry measurement of this robot.
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
  * @brief Construct the quadratic data matrix Q in the local PGO problem
     f(X) = 0.5<Q, XtX> + <X, G>
  */
  void constructQMatrix();

  /**
   * @brief Construct the cost matrix G in the local PGO problem
      f(X) = 0.5<Q, XtX> + <X, G>
   * @param poseDict: a Map that contains the public pose values from the neighbors
   * @return true if the data matrix is computed successfully
   */
  bool constructGMatrix(const PoseDict &poseDict);

  /**
   * @brief initialize local trajectory estimate
   */
  void localInitialization();

  /**
  Optimize pose graph by calling optimize().
  This function will run in a separate thread.
  */
  void runOptimizationLoop();

  /**
   * @brief Find and return any shared measurement with the specified neighbor pose
   * @return
   */
  RelativeSEMeasurement &findSharedLoopClosureWithNeighbor(const PoseID &nID);

  /**
   * @brief Find and return the specified shared measurement
   * @param srcRobotID
   * @param srcPoseID
   * @param dstRobotID
   * @param dstPoseID
   * @return
   */
  RelativeSEMeasurement &findSharedLoopClosure(const PoseID &srcID, const PoseID &dstID);

  /**
   * @brief Return true if should update loop closure weights
   * @return bool
   */
  bool shouldUpdateLoopClosureWeights() const;

  /**
   * @brief Update loop closure weights.
   */
  void updateLoopClosuresWeights();

  /**
   * @brief Compute the ratio of loop closure weights that have converged (assuming GNC_TLS)
   * @return ratio
   */
  double computeConvergedLoopClosureRatio();

 private:
  // Stores the auxiliary variables from neighbors (only used in acceleration)
  PoseDict neighborAuxPoseDict;

  // Auxiliary scalar used in acceleration
  double gamma{};

  // Auxiliary scalar used in acceleration
  double alpha{};

  // Auxiliary variable used in acceleration
  Matrix Y;

  // Auxiliary variable used in acceleration
  Matrix V;

  // Save previous iteration (for restarting)
  Matrix XPrev;

  void updateGamma();

  void updateAlpha();

  /**
   * @brief Update X variable
   * @param doOptimization Whether this agent is selected to perform optimization
   * @param acceleration true to use acceleration
   * @return true if update is successful
   */
  bool updateX(bool doOptimization = false, bool acceleration = false);

  void updateY();

  void updateV();

  void resetTeamStatus();

  /**
   * @brief Return True is the given measurement is already present
   * @param m
   * @param measurements
   * @return
   */
  static bool isDuplicateMeasurement(const RelativeSEMeasurement &m, const vector<RelativeSEMeasurement> &measurements);
};

}  // namespace DPGO

#endif
