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
    : id_(id), r_(r), d_(d), n_(0) {
  CHECK(r >= d);
  empty();
}

PoseGraph::~PoseGraph() {
  empty();
}

void PoseGraph::empty() {
  // Reset this pose graph to be empty
  n_ = 0;
  odometry_.clear();
  private_lcs_.clear();
  shared_lcs_.clear();
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();
  nbr_robot_ids_.clear();
  clearNeighborPoses();
  clearDataMatrices();
}

void PoseGraph::reset() {
  clearNeighborPoses();
  clearDataMatrices();
}

void PoseGraph::clearNeighborPoses() {
  neighbor_poses_.clear();
  G_.reset();  // Clearing neighbor poses requires re-computing linear matrix
}

unsigned int PoseGraph::numMeasurements() const {
  return numOdometry() + numPrivateLoopClosures() + numSharedLoopClosures();
}

void PoseGraph::setMeasurements(const std::vector<RelativeSEMeasurement> &measurements) {
  // Reset this pose graph to be empty
  empty();
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
  if (findMeasurement(src_id, dst_id))
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
  if (findMeasurement(src_id, dst_id))
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
  if (findMeasurement(src_id, dst_id))
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
  G_.reset();  // Setting neighbor poses requires re-computing linear matrix
}

bool PoseGraph::hasNeighbor(unsigned int robot_id) const {
  return nbr_robot_ids_.find(robot_id) != nbr_robot_ids_.end();
}

bool PoseGraph::hasNeighborPose(const PoseID &pose_id) const {
  return nbr_shared_pose_ids_.find(pose_id) != nbr_shared_pose_ids_.end();
}

RelativeSEMeasurement *PoseGraph::findMeasurement(const PoseID &srcID, const PoseID &dstID) {
  for (auto &m : odometry_) {
    if (m.r1 == srcID.robot_id && m.p1 == srcID.frame_id && dstID.robot_id == m.r2 && dstID.frame_id == m.p2) {
      return &m;
    }
  }
  for (auto &m : private_lcs_) {
    if (m.r1 == srcID.robot_id && m.p1 == srcID.frame_id && dstID.robot_id == m.r2 && dstID.frame_id == m.p2) {
      return &m;
    }
  }
  for (auto &m : shared_lcs_) {
    if (m.r1 == srcID.robot_id && m.p1 == srcID.frame_id && dstID.robot_id == m.r2 && dstID.frame_id == m.p2) {
      return &m;
    }
  }
  return nullptr;
}

std::vector<RelativeSEMeasurement *> PoseGraph::writableLoopClosures() {
  std::vector<RelativeSEMeasurement *> output;
  for (auto &m : private_lcs_) {
    output.push_back(&m);
  }
  for (auto &m : shared_lcs_) {
    output.push_back(&m);
  }
  return output;
}

PoseGraph::Statistics PoseGraph::statistics() const {
  // Currently, this function is only meaningful for GNC_TLS
  double totalCount = 0;
  double acceptCount = 0;
  double rejectCount = 0;
  // TODO: specify tolerance for rejected and accepted loop closures
  for (const auto &m : private_lcs_) {
    // if (m.fixedWeight) continue;
    if (m.weight == 1) {
      acceptCount += 1;
    } else if (m.weight == 0) {
      rejectCount += 1;
    }
    totalCount += 1;
  }
  for (const auto &m : shared_lcs_) {
    // if (m.fixedWeight) continue;
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
  statistics.undecided_loop_closures = totalCount - acceptCount - rejectCount;

  return statistics;
}

const SparseMatrix &PoseGraph::quadraticMatrix() {
  if (!Q_.has_value())
    constructQ();
  CHECK(Q_.has_value());
  return Q_.value();
}

void PoseGraph::clearQuadraticMatrix() {
  Q_.reset();
  precon_.reset();  // Also clear the preconditioner since it depends on Q
}

const Matrix &PoseGraph::linearMatrix() {
  if (!G_.has_value())
    constructG();
  CHECK(G_.has_value());
  return G_.value();
}

void PoseGraph::clearLinearMatrix() {
  G_.reset();
}

bool PoseGraph::constructDataMatrices() {
  if (!Q_.has_value() && !constructQ())
    return false;
  if (!G_.has_value() && !constructG())
    return false;
  return true;
}

void PoseGraph::clearDataMatrices() {
  clearQuadraticMatrix();
  clearLinearMatrix();
}

bool PoseGraph::constructQ() {
  timer_.tic();
  std::vector<RelativeSEMeasurement> privateMeasurements = odometry_;
  privateMeasurements.insert(privateMeasurements.end(), private_lcs_.begin(), private_lcs_.end());

  // Initialize Q with private measurements
  SparseMatrix QLocal = constructConnectionLaplacianSE(privateMeasurements);

  // Initialize relative SE matrix in homogeneous form
  Matrix T = Matrix::Zero(d_ + 1, d_ + 1);

  // Initialize aggregate weight matrix
  Matrix Omega = Matrix::Zero(d_ + 1, d_ + 1);

  // Shared (inter-robot) measurements only affect the diagonal blocks
  Matrix QSharedDiag(d_ + 1, (d_ + 1) * n_);
  QSharedDiag.setZero();

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
      int idx = (int) m.p1;
      Matrix W = T * Omega * T.transpose();
      QSharedDiag.block(0, idx * (d_ + 1), d_ + 1, d_ + 1) += W;

    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      CHECK(m.r2 == id_);
      // Modify quadratic cost
      int idx = (int) m.p2;
      QSharedDiag.block(0, idx * (d_ + 1), d_ + 1, d_ + 1) += Omega;
    }
  }

  // Convert QSharedDiag to a sparse matrix
  std::vector<Eigen::Triplet<double>> tripletList;
  tripletList.reserve((d_ + 1) * (d_ + 1) * n_);
  for (unsigned idx = 0; idx < n_; ++idx) {
    unsigned row_base = idx * (d_ + 1);
    unsigned col_base = row_base;
    for (unsigned r = 0; r < d_ + 1; ++r) {
      for (unsigned c = 0; c < d_ + 1; ++c) {
        double val = QSharedDiag(r, col_base + c);
        tripletList.emplace_back(row_base + r, col_base + c, val);
      }
    }
  }
  SparseMatrix QShared(QLocal.rows(), QLocal.cols());
  QShared.setFromTriplets(tripletList.begin(), tripletList.end());

  Q_.emplace(QLocal + QShared);
  ms_construct_Q_ = timer_.toc();
  // LOG(INFO) << "Robot " << id_ << " construct Q ms: " << ms_construct_Q_;
  return true;
}

bool PoseGraph::constructG() {
  timer_.tic();
  unsigned d = d_;
  Matrix G(r_, (d_ + 1) * n_);
  G.setZero();
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
      auto pair = neighbor_poses_.find(nID);
      if (pair == neighbor_poses_.end()) {
        LOG(WARNING) << "Robot " << id_ << " cannot find neighbor pose "
                     << nID.robot_id << ", " << nID.frame_id;
        return false;
      }
      Matrix Xj = pair->second.pose();
      int idx = (int) m.p1;
      // Modify linear cost
      Matrix L = -Xj * Omega * T.transpose();
      G.block(0, idx * (d_ + 1), r_, d_ + 1) += L;
    } else {
      // Second pose belongs to this robot
      // Hence, this is an incoming edge in the pose graph
      CHECK(m.r2 == id_);

      // Read neighbor's pose
      const PoseID nID(m.r1, m.p1);
      auto pair = neighbor_poses_.find(nID);
      if (pair == neighbor_poses_.end()) {
        LOG(WARNING) << "Robot " << id_ << " cannot find neighbor pose "
                     << nID.robot_id << ", " << nID.frame_id;
        return false;
      }
      Matrix Xi = pair->second.pose();
      int idx = (int) m.p2;
      // Modify linear cost
      Matrix L = -Xi * T * Omega;
      G.block(0, idx * (d_ + 1), r_, d_ + 1) += L;
    }
  }
  G_.emplace(G);
  ms_construct_G_ = timer_.toc();
  // LOG(INFO) << "Robot " << id_ << " construct G ms: " << ms_construct_G_;
  return true;
}

bool PoseGraph::hasPreconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  return precon_.has_value();
}
/**
 * @brief Get preconditioner
 * @return
 */
const CholmodSolverPtr & PoseGraph::preconditioner() {
  if (!precon_.has_value())
    constructPreconditioner();
  CHECK(precon_.has_value());
  return precon_.value();
}

bool PoseGraph::constructPreconditioner() {
  timer_.tic();
  // Update preconditioner
  SparseMatrix P = quadraticMatrix();
  for (int i = 0; i < P.rows(); ++i) {
    P.coeffRef(i, i) += 1e-1;
  }
  auto solver = std::make_shared<CholmodSolver>();
  solver->compute(P);
  if (solver->info() != Eigen::ComputationInfo::Success)
    return false;
  precon_.emplace(solver);
  ms_construct_precon_ = timer_.toc();
  //LOG(INFO) << "Construct precon ms: " << ms_construct_precon_;
  return true;
}

void PoseGraph::removeNeighbor(unsigned int robot_id) {
  if (!hasNeighbor(robot_id))
    return;
  nbr_robot_ids_.erase(nbr_robot_ids_.find(robot_id));

  // clear cached data
  clearDataMatrices();
  
  // Remove all shared loop closures with this robot
  int num_lcs_removed = 0;
  auto it = shared_lcs_.begin();
  while (it != shared_lcs_.end()) {
    if (it->r1 == robot_id || it->r2 == robot_id) {
      it = shared_lcs_.erase(it);
      num_lcs_removed++;
    } else {
      it++;
    }
  }
  LOG(INFO) << "Removed " << num_lcs_removed << " loop closures with robot " << robot_id;

  // Update records of public poses from myself and my neighbors
  updatePublicPoseIDs();
}

void PoseGraph::updatePublicPoseIDs() {
  local_shared_pose_ids_.clear();
  nbr_shared_pose_ids_.clear();

  for (const auto& m: shared_lcs_) {
    if (m.r1 == id_) {
      CHECK(m.r2 != id_);
      local_shared_pose_ids_.emplace(m.r1, m.p1);
      nbr_shared_pose_ids_.emplace(m.r2, m.p2);
    } else {
      CHECK(m.r2 == id_);
      local_shared_pose_ids_.emplace(m.r2, m.p2);
      nbr_shared_pose_ids_.emplace(m.r1, m.p1);
    }
  }
}

}