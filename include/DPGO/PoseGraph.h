/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGO_INCLUDE_DPGO_POSEGRAPH_H_
#define DPGO_INCLUDE_DPGO_POSEGRAPH_H_

#include "DPGO_types.h"
#include "manifold/Poses.h"
#include "RelativeSEMeasurement.h"
#include <set>

namespace DPGO {

/**
 * @brief A pose graph class representing the local optimization problem in distributed PGO.
 */
class PoseGraph {
 public:
  /**
   * @brief Store statistics for the current pose graph
   */
  class Statistics {
   public:
    Statistics() : total_loop_closures(0), accept_loop_closures(0), reject_loop_closures(0) {}
    double total_loop_closures;
    double accept_loop_closures;
    double reject_loop_closures;
  };
  /**
   * @brief
   * @param id
   * @param r
   * @param d
   */
  PoseGraph(unsigned int id, unsigned int r, unsigned int d);
  /**
   * @brief Get dimension
   * @return
   */
  unsigned int d() const { return d_; }
  /**
   * @brief Get relaxation rank
   * @return
   */
  unsigned int r() const { return r_; }
  /**
   * @brief Get number of poses
   * @return
   */
  unsigned int n() const { return n_; }
  /**
   * @brief Clear all contents in the pose graph
   */
  void clear();
  /**
   * @brief Set measurements for this pose graph
   * @param measurements
   */
  void setMeasurements(const std::vector<RelativeSEMeasurement> &measurements);
  /**
   * @brief Add a single measurement to this pose graph
   * @param m
   */
  void addMeasurement(const RelativeSEMeasurement& m);
  /**
   * @brief Add odometry edge
   * @param factor
   */
  void addOdometry(const RelativeSEMeasurement &factor);
  /**
   * @brief Add private loop closure
   * @param factor
   */
  void addPrivateLoopClosure(const RelativeSEMeasurement &factor);
  /**
   * @brief Add shared loop closure
   * @param factor
   */
  void addSharedLoopClosure(const RelativeSEMeasurement &factor);
  /**
   * @brief Return all odometry measurements
   * @return
   */
  std::vector<RelativeSEMeasurement> &odometry() { return odometry_; }
  /**
   * @brief Return a writable reference to the list of private loop closures
   * @return
   */
  std::vector<RelativeSEMeasurement> &privateLoopClosures() { return private_lcs_; }
  /**
   * @brief Return a writable reference to the list of shared loop closures
   * @return
   */
  std::vector<RelativeSEMeasurement> &sharedLoopClosures() { return shared_lcs_; }
  /**
   * @brief Return all inter-robot loop closures with the specified neighbor
   * @param neighbor_id
   * @return
   */
  std::vector<RelativeSEMeasurement> sharedLoopClosuresWithRobot(unsigned neighbor_id) const;
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
   * @brief Return a vector of all measurements
   * @return
   */
  std::vector<RelativeSEMeasurement> measurements() const;
  /**
   * @brief Return a vector of all LOCAL measurements (i.e., without inter-robot loop closures)
   * @return
   */
  std::vector<RelativeSEMeasurement> localMeasurements() const;
  /**
   * @brief Set neighbor poses
   * @param pose_dict
   */
  void setNeighborPoses(const PoseDict &pose_dict);
  /**
   * @brief Initialize for optimization
   * @return
   */
  bool initialize();
  /**
   * @brief Return true if the graph is ready for optimization
   * @return
   */
  bool isInitialized() const { return initialized_; }
  /**
   * @brief Get quadratic cost matrix. Pose graph must be initialized.
   * @return
   */
  SparseMatrix quadraticMatrix() const {
    if (!isInitialized())
      throw std::runtime_error("Attempt to get quadratic matrix from uninitialized pose graph.");
    return Q_;
  }
  /**
   * @brief Get linear cost matrix. Pose graph must be initialized.
   * @return
   */
  SparseMatrix linearMatrix() const {
    if (!isInitialized())
      throw std::runtime_error("Attempt to get quadratic matrix from uninitialized pose graph.");
    return G_;
  }
  /**
   * @brief Get the set of my pose IDs that are shared with other robots
   * @return
   */
  PoseSet myPublicPoseIDs() const { return local_shared_pose_ids_; }
  /**
   * @brief Get the set of Pose IDs that my neighbors need to share with me
   * @return
   */
  PoseSet neighborPublicPoseIDs() const { return nbr_shared_pose_ids_; }
  /**
   * @brief Get the set of neighbor robot IDs that share inter-robot loop closures with me
   * @return
   */
  std::set<unsigned> neighborIDs() const { return nbr_robot_ids_; }
  /**
   * @brief Return the number of neighbors
   * @return
   */
  size_t numNeighbors() const { return nbr_robot_ids_.size(); }
  /**
   * @brief Return true if the input robot is a neighbor (i.e., share inter-robot loop closure)
   * @param robot_id
   * @return
   */
  bool hasNeighbor(unsigned int robot_id) const;
  /**
   * @brief Return true if the given neighbor pose ID is required by me
   * @param pose_id
   * @return
   */
  bool hasNeighborPose(const PoseID &pose_id) const;


  /**
   * @brief Compute statistics for the current pose graph
   * @return
   */
  Statistics statistics() const;

 protected:

  // ID associated with this agent
  const unsigned int id_;

  // Problem dimensions
  unsigned int r_, d_, n_;

  // Ready for optimization
  bool initialized_;

  // Store odometry measurement of this robot
  std::vector<RelativeSEMeasurement> odometry_;

  // Store private loop closures of this robot
  std::vector<RelativeSEMeasurement> private_lcs_;

  // Store shared loop closure measurements
  std::vector<RelativeSEMeasurement> shared_lcs_;

  // Store the set of public poses that need to be sent to other robots
  PoseSet local_shared_pose_ids_;

  // Store the set of public poses needed from other robots
  PoseSet nbr_shared_pose_ids_;

  // Store the set of neighboring agents
  std::set<unsigned> nbr_robot_ids_;

  // Store public poses from neighbors
  PoseDict neighbor_poses_;

  // Quadratic matrix in cost function
  SparseMatrix Q_;

  // Linear matrix in cost function
  SparseMatrix G_;
  /**
   * @brief Construct the quadratic cost matrix
   * @return
   */
  bool constructQ();
  /**
   * @brief Construct the linear cost matrix
   * @return
   */
  bool constructG();

};

}

#endif //DPGO_INCLUDE_DPGO_POSEGRAPH_H_
