/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "DPGO/PoseGraph.h"
#include "DPGO/DPGO_utils.h"
#include <glog/logging.h>

namespace DPGO {

PoseGraph::PoseGraph(unsigned int id, unsigned int r, unsigned int d)
    : id_(id), r_(r), d_(d), n_(0), initialized_(false) {
  CHECK(r >= d);
  clear();
}

void PoseGraph::clear() {
  n_ = 0;
  odometry_.clear();
  private_lcs_.clear();
  shared_lcs_.clear();
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_robot_ids_.clear();
  neighbor_poses_.clear();
}

void PoseGraph::setMeasurements(const std::vector<RelativeSEMeasurement> &measurements) {
  // Reset this pose graph to be empty
  clear();
  for (const auto &m : measurements)
    addMeasurement(m);
}

void PoseGraph::addMeasurement(const RelativeSEMeasurement &m) {
  if (m.r1 != id_ && m.r2 != id_) {
    LOG(WARNING) << "Input contains irrelevant edges! \n" << m;
    return;
  }
  if (m.r1 == id_ && m.r2 == id_) {
    if (m.p1 + 1 == m.p2)
      addOdometry(m);
    else
      addPrivateLoopClosure(m);
  } else {
    addSharedLoopClosure(m);
  }
}

void PoseGraph::addOdometry(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (findMeasurement(odometry_, src_id, dst_id))
    return;

  // Check that this is an odometry measurement
  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.p1 + 1 == factor.p2);
  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  n_ = std::max(n_, (unsigned int) factor.p2 + 1);
  odometry_.push_back(factor);
}

void PoseGraph::addPrivateLoopClosure(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (findMeasurement(private_lcs_, src_id, dst_id))
    return;

  CHECK(factor.r1 == id_);
  CHECK(factor.r2 == id_);
  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  // update number of poses
  n_ = std::max(n_, (unsigned int) std::max(factor.p1 + 1, factor.p2 + 1));
  private_lcs_.push_back(factor);
}

void PoseGraph::addSharedLoopClosure(const RelativeSEMeasurement &factor) {
  // Check for duplicate inter-robot loop closure
  const PoseID src_id(factor.r1, factor.p1);
  const PoseID dst_id(factor.r2, factor.p2);
  if (findMeasurement(shared_lcs_, src_id, dst_id))
    return;

  CHECK(factor.R.rows() == d_ && factor.R.cols() == d_);
  CHECK(factor.t.rows() == d_ && factor.t.cols() == 1);
  if (factor.r1 == id_) {
    CHECK(factor.r2 != id_);
    n_ = std::max(n_, (unsigned int) factor.p1 + 1);
    local_shared_pose_ids_.emplace(factor.r1, factor.p1);
    nbr_shared_pose_ids_.emplace(factor.r2, factor.p2);
    nbr_robot_ids_.insert(factor.r2);
  } else {
    CHECK(factor.r2 == id_);
    n_ = std::max(n_, (unsigned int) factor.p2 + 1);
    local_shared_pose_ids_.emplace(factor.r2, factor.p2);
    nbr_shared_pose_ids_.emplace(factor.r1, factor.p1);
    nbr_robot_ids_.insert(factor.r1);
  }

  shared_lcs_.push_back(factor);
}

std::vector<RelativeSEMeasurement> PoseGraph::sharedLoopClosuresWithRobot(unsigned int neighbor_id) const {
  std::vector<RelativeSEMeasurement> result;
  for (const auto &m : shared_lcs_) {
    if (m.r1 == neighbor_id || m.r2 == neighbor_id)
      result.emplace_back(m);
  }
  return result;
}

std::vector<RelativeSEMeasurement> PoseGraph::measurements() const {
  std::vector<RelativeSEMeasurement> measurements = odometry_;
  measurements.insert(measurements.end(), private_lcs_.begin(), private_lcs_.end());
  measurements.insert(measurements.end(), shared_lcs_.begin(), shared_lcs_.end());
  return measurements;
}

std::vector<RelativeSEMeasurement> PoseGraph::localMeasurements() const {
  std::vector<RelativeSEMeasurement> measurements = odometry_;
  measurements.insert(measurements.end(), private_lcs_.begin(), private_lcs_.end());
  return measurements;
}

void PoseGraph::setNeighborPoses(const PoseDict &pose_dict) {
  neighbor_poses_ = pose_dict;
}

bool PoseGraph::initialize() {
  constructQ();
  if (constructG())
    initialized_ = true;
  else
    initialized_ = false;
  return initialized_;
}

bool PoseGraph::hasNeighbor(unsigned int robot_id) const {
  return nbr_robot_ids_.find(robot_id) != nbr_robot_ids_.end();
}

bool PoseGraph::hasNeighborPose(const PoseID &pose_id) const {
  return nbr_shared_pose_ids_.find(pose_id) != nbr_shared_pose_ids_.end();
}

RelativeSEMeasurement *PoseGraph::findMeasurement(std::vector<RelativeSEMeasurement> &measurements,
                                                  const PoseID &srcID,
                                                  const PoseID &dstID) {
  for (auto &m : measurements) {
    if (m.r1 == srcID.robot_id && m.p1 == srcID.frame_id && dstID.robot_id == m.r2 && dstID.frame_id == m.p2) {
      return &m;
    }
  }
  return nullptr;
}

PoseGraph::Statistics PoseGraph::statistics() const {
  // Currently, this function is only meaningful for GNC_TLS
  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  // TODO: specify tolerance for rejected and accepted loop closures
  for (const auto &m : private_lcs_) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  for (const auto &m : shared_lcs_) {
    if (m.isKnownInlier) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }

  PoseGraph::Statistics statistics;
  statistics.total_loop_closures = totalCount;
  statistics.accept_loop_closures = acceptCount;
  statistics.reject_loop_closures = rejectCount;

  return statistics;
}

bool PoseGraph::constructQ() {
  std::vector<RelativeSEMeasurement> privateMeasurements = odometry_;
  privateMeasurements.insert(privateMeasurements.end(), private_lcs_.begin(), private_lcs_.end());

  // Initialize Q with private measurements
  SparseMatrix Q = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d_ + 1, d_ + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d_ + 1, d_ + 1);

  // Go through shared loop closures
  for (const auto &m : shared_lcs_) {
    // Set relative SE matrix (homogeneous form)
    T.block(0, 0, d_, d_) = m.R;
    T.block(0, d_, d_, 1) = m.t;
    T(d_, d_) = 1;

    // Set aggregate weight matrix
    for (unsigned row = 0; row < d_; ++row) {
      Omega(row, row) = m.weight * m.kappa;
    }
    Omega(d_, d_) = m.weight * m.tau;

    if (m.r1 == id_) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      CHECK(m.r2 != id_);

      // Modify quadratic cost
      size_t idx = m.p1;

      Matrix W = T * Omega * T.transpose();

      for (size_t col = 0; col < d_ + 1; ++col) {
        for (size_t row = 0; row < d_ + 1; ++row) {
          Q.coeffRef(idx * (d_ + 1) + row, idx * (d_ + 1) + col) += W(row, col);
        }
      }

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      CHECK(m.r2 == id_);

      // Modify quadratic cost
      size_t idx = m.p2;

      for (size_t col = 0; col < d_ + 1; ++col) {
        for (size_t row = 0; row < d_ + 1; ++row) {
          Q.coeffRef(idx * (d_ + 1) + row, idx * (d_ + 1) + col) +=
              Omega(row, col);
        }
      }
    }
  }

  Q_ = Q;
  return true;
}

bool PoseGraph::constructG() {
  unsigned d = d_;
  unsigned n = n_;
  unsigned r = r_;
  SparseMatrix G(r, (d + 1) * n);
  for (const auto &m : shared_lcs_) {
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

    if (m.r1 == id_) {
      // First pose belongs to this robot
      // Hence, this is an outgoing edge in the pose graph
      CHECK(m.r2 != id_);

      // Read neighbor's pose
      const PoseID nID(m.r2, m.p2);
      auto KVpair = neighbor_poses_.find(nID);
      if (KVpair == neighbor_poses_.end()) {
        printf("constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
               id_, nID.robot_id, nID.frame_id);
        return false;
      }
      Matrix Xj = KVpair->second.pose();

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
      CHECK(m.r2 == id_);

      // Read neighbor's pose
      const PoseID nID(m.r1, m.p1);
      auto KVpair = neighbor_poses_.find(nID);
      if (KVpair == neighbor_poses_.end()) {
        printf("constructGMatrix: robot %u cannot find neighbor pose (%u, %u)\n",
               id_, nID.robot_id, nID.frame_id);
        return false;
      }
      Matrix Xi = KVpair->second.pose();

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
  G_ = G;
  return true;
}

}