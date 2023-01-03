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
#include <DPGO/manifold/Poses.h>
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
#include <glog/logging.h>

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
This class contains parameter settings for PGOAgent
*/
class PGOAgentParameters {
 public:
  // Problem dimension
  unsigned d;

  // Relaxed rank in Riemannian optimization
  unsigned r;

  // Total number of robots
  unsigned numRobots;

  // Run in asynchronous mode
  bool asynchronous;

  // Frequency of optimization loop in asynchronous mode
  double asynchronousOptimizationRate;

  // Riemannian optimization settings for solving local subproblem
  ROptParameters localOptimizationParams;

  // Method to use to initialize single-robot trajectory estimates
  InitializationMethod localInitializationMethod;

  // Cross-robot initialization
  bool multirobotInitialization;

  // Use Nesterov acceleration
  bool acceleration;

  // Interval for fixed (periodic) restart
  unsigned restartInterval;

  // Parameter settings over robust cost functions
  RobustCostParameters robustCostParams;

  // Number of weight updates for robust optimization
  int robustOptNumWeightUpdates;

  // Warm start iterate during robust optimization
  int robustOptNumResets;

  // Number of inner iterations to apply before updating measurement weights during robust optimization
  int robustOptInnerIters;

  // Minimum ratio of converged weights before terminating robust optimization
  double robustOptMinConvergenceRatio;

  // Minimum number of inliers for robust distributed initialization
  unsigned robustInitMinInliers;

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
                     ROptParameters local_opt_params = ROptParameters(),
                     bool accel = false,
                     unsigned restartInt = 30,
                     RobustCostParameters costParams = RobustCostParameters(),
                     int robust_opt_num_weight_updates = 10,
                     int robust_opt_num_resets = 0,
                     int robust_opt_inner_iters = 30,
                     double robust_opt_min_convergence_ratio = 0.8,
                     unsigned robust_init_min_inliers = 2,
                     unsigned maxIters = 500,
                     double changeTol = 5e-3,
                     bool v = false,
                     bool log = false,
                     std::string logDir = "")
      : d(dIn), r(rIn), numRobots(numRobotsIn),
        asynchronous(false),
        asynchronousOptimizationRate(1),
        localOptimizationParams(local_opt_params),
        localInitializationMethod(InitializationMethod::Odometry),
        multirobotInitialization(true),
        acceleration(accel),
        restartInterval(restartInt),
        robustCostParams(costParams),
        robustOptNumWeightUpdates(robust_opt_num_weight_updates),
        robustOptNumResets(robust_opt_num_resets),
        robustOptInnerIters(robust_opt_inner_iters),
        robustOptMinConvergenceRatio(robust_opt_min_convergence_ratio),
        robustInitMinInliers(robust_init_min_inliers),
        maxNumIters(maxIters),
        relChangeTol(changeTol),
        verbose(v),
        logData(log),
        logDirectory(std::move(logDir)) {}

  inline friend std::ostream &operator<<(
      std::ostream &os, const PGOAgentParameters &params) {
    os << "PGOAgent parameters: " << std::endl;
    os << "Dimension: " << params.d << std::endl;
    os << "Relaxation rank: " << params.r << std::endl;
    os << "Number of robots: " << params.numRobots << std::endl;
    os << "Asynchronous: " << params.asynchronous << std::endl;
    os << "Asynchronous optimization rate: " << params.asynchronousOptimizationRate << std::endl;
    os << "Local initialization method: " << InitializationMethodToString(params.localInitializationMethod)
       << std::endl;
    os << "Use multi-robot initialization: " << params.multirobotInitialization << std::endl;
    os << "Use Nesterov acceleration: " << params.acceleration << std::endl;
    os << "Fixed restart interval: " << params.restartInterval << std::endl;
    os << "Robust optimization num weight updates: " << params.robustOptNumWeightUpdates << std::endl;
    os << "Robust optimization num resets: " << params.robustOptNumResets << std::endl;
    os << "Robust optimization inner iterations: " << params.robustOptInnerIters << std::endl;
    os << "Robust optimization weight convergence min ratio: " << params.robustOptMinConvergenceRatio << std::endl;
    os << "Robust initialization minimum inliers: " << params.robustInitMinInliers << std::endl;
    os << "Max iterations: " << params.maxNumIters << std::endl;
    os << "Relative change tol: " << params.relChangeTol << std::endl;
    os << "Verbose: " << params.verbose << std::endl;
    os << "Log data: " << params.logData << std::endl;
    os << "Log directory: " << params.logDirectory << std::endl;
    os << std::endl;
    os << params.localOptimizationParams << std::endl;
    os << std::endl;
    os << params.robustCostParams << std::endl;
    return os;
  }
};

/**
Defines the possible states of a PGOAgent
Each state can only transition to the state below
*/
enum PGOAgentState {

  WAIT_FOR_DATA,  // waiting to receive pose graph

  WAIT_FOR_INITIALIZATION,  // waiting to initialize trajectory estimate

  INITIALIZED,  // trajectory initialized and ready to update

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
  explicit PGOAgentStatus(unsigned id = 0,
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

  /**
   * @brief Constructor
   * @param ID
   * @param params
   */
  PGOAgent(unsigned ID, const PGOAgentParameters &params);

  /**
   * @brief Destructor
   */
  ~PGOAgent();

  /**
   * @brief Set measurements (pose graph) for this agent
   * @param inputOdometry
   * @param inputPrivateLoopClosures
   * @param inputSharedLoopClosures
   */
  void setMeasurements(const std::vector<RelativeSEMeasurement> &inputOdometry,
                       const std::vector<RelativeSEMeasurement> &inputPrivateLoopClosures,
                       const std::vector<RelativeSEMeasurement> &inputSharedLoopClosures);

  /**
   * @brief Add a single measurement to this agent's pose graph. Do nothing if the input factor already exists.
   * @param factor
   */
  void addMeasurement(const RelativeSEMeasurement &factor);

  /**
   * @brief Initialize distributed optimization.
   * @param TInitPtr an optional trajectory estimate in an arbitrary local frame. If the dimension of number of poses
   * of the provided initial guess does not match what is expected, this initial guess will be ignored and this function
   * will perform initialization on its own.
   */
  void initialize(const PoseArray *TInitPtr = nullptr);

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
  Return ID of this robot
  */
  inline unsigned getID() const { return mID; }

  /**
  Return number of poses of this robot
  */
  inline unsigned num_poses() const { return mPoseGraph->n(); }

  /**
  Get dimension
  */
  inline unsigned dimension() const { return mPoseGraph->d(); }

  /**
  Get relaxation rank
  */
  inline unsigned relaxation_rank() const { return mPoseGraph->r(); }

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
   * @brief return true if the status of a neighbor robot is available locally
   * @return
   */
  bool hasNeighborStatus(unsigned neighborID) const {
    return mTeamStatus.find(neighborID) != mTeamStatus.end();
  }
  /**
   * @brief get current state of a neighbor
   * @param neighborID
   * @return
   */
  inline PGOAgentStatus getNeighborStatus(unsigned neighborID) const {
    CHECK(hasNeighborStatus(neighborID));
    return mTeamStatus.at(neighborID);
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
   * Return true if the input robot is a neighbor 
   * (i.e., has inter-robot loop closure with this robot)
   */
  bool hasNeighbor(unsigned neighborID) const;

  /**
  Get vector of neighbor robot IDs.
  */
  std::vector<unsigned> getNeighbors() const;

  /**
   * Remove the specified neighbor.
   * No effect if the input robot is not a neighbor
   */
  void removeNeighbor(unsigned neighborID);

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
  Return trajectory estimate of this robot in global frame, with the first pose
  of robot 0 set to identity
  */
  bool getTrajectoryInGlobalFrame(PoseArray &Trajectory);

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
   * Get a map of all public poses of this robot with the specified neighbor
   */
  bool getSharedPoseDictWithNeighbor(PoseDict &map, unsigned neighborID);

  /**
   * @brief Get a map of all auxiliary variables associated with public poses of this robot
   * @param map
   * @return true if agent is initialized
   */
  bool getAuxSharedPoseDict(PoseDict &map);

  /**
   * Get a map of all auxiliary public poses of this robot with the specified neighbor
   */
  bool getAuxSharedPoseDictWithNeighbor(PoseDict &map, unsigned neighborID);

  /**
   * @brief Helper function to reset internal solution. Currently only for debugging.
   * @param Xin
   */
  void setX(const Matrix &Xin);

  /**
   * @brief Reset internal solution to initial guess X = Xinit.
   */
  void setXToInitialGuess();

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
  void startOptimizationLoop();

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
   * @brief Clear local caches of all neighbors' poses
   */
  void clearNeighborPoses();

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

  // Internal optimization iterate (before rounding)
  LiftedPoseArray X;

  // Parameter settings
  const PGOAgentParameters mParams;

  // Current state of this agent
  PGOAgentState mState;

  // Current status of this agent (to be shared with others)
  PGOAgentStatus mStatus;

  // Robust cost function
  RobustCost mRobustCost;

  // Pointer to pose graph
  std::shared_ptr<PoseGraph> mPoseGraph;

  // Current PGO instance
  unsigned mInstanceNumber;

  // Current global iteration counter (this is only meaningful in synchronous mode)
  unsigned mIterationNumber;

  // Number of inner iterations performed for robust optimization
  int mRobustOptInnerIter;

  // Number of times measurement weights are updated
  int mWeightUpdateCount;

  // Number of times solutions are reset to initial guess
  int mTrajectoryResetCount;

  // Latest local optimization result
  ROPTResult mLocalOptResult;

  // Logging
  PGOLogger mLogger;

  // Store status of peer agents
  std::unordered_map<unsigned, PGOAgentStatus> mTeamStatus;

  // Store if robots are actively participating in optimization
  std::vector<bool> mTeamRobotActive;

  // Request to publish public poses
  bool mPublishPublicPosesRequested = false;

  // Request to publish in asynchronous mode
  bool mPublishAsynchronousRequested = false;

  // Request to terminate optimization thread
  bool mEndLoopRequested = false;

  // Initial iterate
  std::optional<LiftedPoseArray> XInit;

  // Initial solution TInit = [R1 t1 ... Rn tn] in an arbitrary coordinate frame
  std::optional<PoseArray> TLocalInit;

  // Lifting matrix shared by all agents
  std::optional<Matrix> YLift;

  // Anchor matrix shared by all agents
  std::optional<LiftedPose> globalAnchor;

  // This dictionary stores poses owned by other robots that is connected to this robot by loop closure
  PoseDict neighborPoseDict;

  // Implement locking to synchronize read & write of trajectory estimate
  mutex mPosesMutex;

  // Implement locking to synchronize read & write of shared poses from neighbors
  mutex mNeighborPosesMutex;

  // Implement locking on measurements
  mutex mMeasurementsMutex;

  // Thread that runs optimization loop in asynchronous mode
  std::unique_ptr<thread> mOptimizationThread;

  /**
   * @brief Reset variables used in Nesterov acceleration
   */
  void initializeAcceleration();
  /**
   * @brief initialize local trajectory estimate in an arbitrary local frame
   * @return true if local initialization is successful
   */
  bool initializeLocalTrajectory();
  /**
   * @brief Initialize this robot's trajectory estimate in the global frame
   * @param T_world_robot d+1 by d+1 transformation from robot (local) frame to the world frame
   */
  void initializeInGlobalFrame(const Pose &T_world_robot);
  /**
   * @brief Compute a robust relative transform estimate between this robot and neighbor robot, using a two-stage method
   * which first perform robust single rotation averaging, and then performs translation averaging on the inlier set.
   * @param neighborID
   * @param poseDict
   * @param T_world_robot output transformation from current local (robot) frame to world frame
   * @return true if transformation is computed successfully
   */
  bool computeRobustNeighborTransformTwoStage(unsigned neighborID, const PoseDict &poseDict, Pose *T_world_robot);
  /**
   * @brief Compute a robust relative transform estimate between this robot and neighbor robot, by solving a robust single
   * pose averaging problem using GNC.
   * @param neighborID
   * @param poseDict
   * @param T_world_robot output transformation from current local (robot) frame to world frame
   * @return true if transformation is computed successfully
   */
  bool computeRobustNeighborTransform(unsigned neighborID, const PoseDict &poseDict, Pose *T_world_robot);
  /**
   * @brief Spawn a separate thread that optimizes the local pose graph in a loop
   */
  void runOptimizationLoop();
  /**
   * @brief Return true if should update loop closure weights
   * @return bool
   */
  bool shouldUpdateMeasurementWeights() const;
  /**
   * @brief Update loop closure weights.
   */
  void updateMeasurementWeights();
  /**
   * @brief Compute the residual of a measurement (square root of weighted square error)
   * @param measurement The measurement to evaluate
   * @param residual The output residual
   * @return true if computation is successful
   */
  bool computeMeasurementResidual(const RelativeSEMeasurement &measurement,
                                  double *residual) const;
  /**
   * @brief Set weight for measurement in the pose graph.
   * @param src_ID
   * @param dst_ID
   * @param weight
   * @param fixed_weight True if the weight is fixed (i.e. cannot be changed by GNC)
   * @return false if the specified public measurement does not exist
   */
  bool setMeasurementWeight(const PoseID &src_ID, const PoseID &dst_ID,
                            double weight, bool fixed_weight = false);
  /**
   * @brief Return true if the robot is initialized in global frame
   */
  bool isRobotInitialized(unsigned robot_id) const;
  /**
   * @brief Return true if the robot is currently active
   */
  bool isRobotActive(unsigned robot_id) const;
  /**
   * @brief Set robot to be active 
   */
  void setRobotActive(unsigned robot_id, bool active = true);

 private:
  // Stores the auxiliary variables from neighbors (only used in acceleration)
  PoseDict neighborAuxPoseDict;

  // Auxiliary scalar used in acceleration
  double gamma;

  // Auxiliary scalar used in acceleration
  double alpha;

  // Auxiliary variable used in acceleration
  LiftedPoseArray Y;

  // Auxiliary variable used in acceleration
  LiftedPoseArray V;

  // Save previous iteration (for restarting)
  LiftedPoseArray XPrev;

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
  /**
   * @brief Compute the relative transformation to a neighboring robot using a single inter-robot loop closure
   * @param measurement
   * @param neighbor_pose
   * @return
   */
  Pose computeNeighborTransform(const RelativeSEMeasurement &measurement, const LiftedPose &neighbor_pose);
};

}  // namespace DPGO

#endif
