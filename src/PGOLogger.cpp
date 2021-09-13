/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */
#include <cassert>
#include <DPGO/PGOLogger.h>
#include <Eigen/Geometry>
#include <utility>

namespace DPGO {

PGOLogger::PGOLogger(std::string logDir) : logDirectory(std::move(logDir)) {}

PGOLogger::~PGOLogger() = default;

void PGOLogger::logMeasurements(std::vector<RelativeSEMeasurement> &measurements, const std::string &filename) {
  if (measurements.empty()) return;

  std::ofstream file;
  file.open(logDirectory + filename);
  if (!file.is_open()) return;

  size_t d = measurements[0].R.rows();
  if (d == 2) return;

  // Insert header row
  file << "robot_src,pose_src,robot_dst,pose_dst,qx,qy,qz,qw,tx,ty,tz,kappa,tau,is_known_inlier,weight\n";

  for (RelativeSEMeasurement m: measurements) {
    // Convert rotation matrix to quaternion
    Eigen::Matrix3d R = m.R;
    Eigen::Quaternion<double> quat(R);
    file << m.r1 << ",";
    file << m.p1 << ",";
    file << m.r2 << ",";
    file << m.p2 << ",";
    file << quat.x() << ",";
    file << quat.y() << ",";
    file << quat.z() << ",";
    file << quat.w() << ",";
    file << m.t(0) << ",";
    file << m.t(1) << ",";
    file << m.t(2) << ",";
    file << m.kappa << ",";
    file << m.tau << ",";
    file << m.isKnownInlier << ",";
    file << m.weight << "\n";
  }

  file.close();
}

void PGOLogger::logTrajectory(unsigned int d, unsigned int n, const Matrix &T, const std::string &filename) {
  if (d == 2) return;
  assert(T.rows() == d);
  assert(T.cols() == (d + 1) * n);
  std::ofstream file;
  file.open(logDirectory + filename);
  if (!file.is_open()) return;

  // Insert header row
  file << "pose_index,qx,qy,qz,qw,tx,ty,tz\n";

  for (size_t i = 0; i < n; ++i) {
    Eigen::Matrix3d R = T.block(0, i * (d + 1), d, d);
    Eigen::Quaternion<double> quat(R);
    Matrix t = T.block(0, i * (d + 1) + d, d, 1);
    file << i << ",";
    file << quat.x() << ",";
    file << quat.y() << ",";
    file << quat.z() << ",";
    file << quat.w() << ",";
    file << t(0) << ",";
    file << t(1) << ",";
    file << t(2) << "\n";
  }

  file.close();
}

Matrix PGOLogger::loadTrajectory(const std::string &filename) {
  std::ifstream infile(logDirectory + filename);
  std::cout << "Loading trajectory from " << logDirectory + filename << "..." << std::endl;
  if (!infile.is_open()) {
    std::cout << "Could not open specified file!" << std::endl;
    return Matrix(0, 0);
  }

  std::unordered_map<uint32_t, Matrix> Tmap;

  // Scalars that will be populated
  uint32_t pose_id = 0;
  uint32_t num_poses = 0;
  double qx, qy, qz, qw;
  double tx, ty, tz;

  std::string line;
  std::string token;

  // Skip first line (headers)
  std::getline(infile, line);

  // Iterate over remaining lines
  while (std::getline(infile, line)) {
    std::istringstream ss(line);
    num_poses++;

    std::getline(ss, token, ',');
    pose_id = std::stoi(token);

    std::getline(ss, token, ',');
    qx = std::stod(token);
    std::getline(ss, token, ',');
    qy = std::stod(token);
    std::getline(ss, token, ',');
    qz = std::stod(token);
    std::getline(ss, token, ',');
    qw = std::stod(token);
    Eigen::Quaternion<double> quat(qw, qx, qy, qz);
    quat.normalize();

    std::getline(ss, token, ',');
    tx = std::stod(token);
    std::getline(ss, token, ',');
    ty = std::stod(token);
    std::getline(ss, token, ',');
    tz = std::stod(token);
    Eigen::Vector3d tVec;
    tVec << tx, ty, tz;

    Matrix Ti(3, 4);
    Ti.block(0, 0, 3, 3) = quat.toRotationMatrix();
    Ti.block(0, 3, 3, 1) = tVec;
    Tmap.emplace(pose_id, Ti);
  }

  Matrix T = Matrix(3, 4 * num_poses);
  for (unsigned i = 0; i < num_poses; ++i) {
    T.block(0, 4 * i, 3, 4) = Tmap.at(i);
  }

  std::cout << "Loaded " << num_poses << " poses." << std::endl;
  return T;
}

std::vector<RelativeSEMeasurement> PGOLogger::loadMeasurements(const std::string &filename, bool load_weight) {
  std::vector<RelativeSEMeasurement> measurements;
  std::cout << "Loading measurements from " << logDirectory + filename << "..." << std::endl;
  std::ifstream infile(logDirectory + filename);

  if (!infile.is_open()) {
    std::cout << "Could not open specified file!" << std::endl;
    return measurements;
  }

  // Scalars that will be filled upon reading each measurement
  uint32_t robot_src, robot_dst, pose_src, pose_dst;
  double qx, qy, qz, qw;
  double tx, ty, tz;
  double kappa, tau, weight;
  bool is_known_inlier;

  std::string line;
  std::string token;

  // Skip first line (headers)
  std::getline(infile, line);

  // Iterate over remaining lines
  while (std::getline(infile, line)) {
    std::istringstream ss(line);

    std::getline(ss, token, ',');
    robot_src = std::stoi(token);
    std::getline(ss, token, ',');
    pose_src = std::stoi(token);
    std::getline(ss, token, ',');
    robot_dst = std::stoi(token);
    std::getline(ss, token, ',');
    pose_dst = std::stoi(token);

    std::getline(ss, token, ',');
    qx = std::stod(token);
    std::getline(ss, token, ',');
    qy = std::stod(token);
    std::getline(ss, token, ',');
    qz = std::stod(token);
    std::getline(ss, token, ',');
    qw = std::stod(token);
    Eigen::Quaternion<double> quat(qw, qx, qy, qz);
    quat.normalize();

    std::getline(ss, token, ',');
    tx = std::stod(token);
    std::getline(ss, token, ',');
    ty = std::stod(token);
    std::getline(ss, token, ',');
    tz = std::stod(token);
    Eigen::Vector3d tVec;
    tVec << tx, ty, tz;

    std::getline(ss, token, ',');
    kappa = std::stod(token);
    std::getline(ss, token, ',');
    tau = std::stod(token);
    std::getline(ss, token, ',');
    is_known_inlier = std::stoi(token);
    std::getline(ss, token, ',');
    weight = std::stod(token);

    RelativeSEMeasurement m(robot_src, robot_dst, pose_src, pose_dst,
                            quat.toRotationMatrix(), tVec,
                            kappa, tau);
    m.isKnownInlier = is_known_inlier;
    if (load_weight)
      m.weight = weight;

    measurements.push_back(m);
  }

  printf("Loaded %zu measurements.\n", measurements.size());
  return measurements;
}

}