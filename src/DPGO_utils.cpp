/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_utils.h>
#include <DPGO/DPGO_robust.h>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <boost/math/distributions/chi_squared.hpp>
#include <glog/logging.h>

namespace DPGO {

std::string ROptParameters::ROptMethodToString(ROptParameters::ROptMethod method) {
  switch (method) {
    case ROptParameters::ROptMethod::RTR: {
      return "RTR";
    }
    case ROptParameters::ROptMethod::RGD: {
      return "RGD";
    }
  }
  return "";
}

std::string InitializationMethodToString(InitializationMethod method) {
  switch (method) {
    case InitializationMethod::Odometry: {
      return "Odometry";
    }
    case InitializationMethod::Chordal: {
      return "Chordal";
    }
    case InitializationMethod::GNC_TLS: {
      return "GNC_TLS";
    }
  }
  return "";
}

void SimpleTimer::tic() {
  t_start = std::chrono::high_resolution_clock::now();
}

double SimpleTimer::toc() {
  t_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> t_elapsed(0);
  t_elapsed = t_end - t_start;
  return t_elapsed.count();
}

std::chrono::time_point<std::chrono::high_resolution_clock> SimpleTimer::Tic() {
  return std::chrono::high_resolution_clock::now();
}

double SimpleTimer::Toc(const std::chrono::time_point<std::chrono::high_resolution_clock> &start_time) {
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::milli> t_elapsed(0);
  t_elapsed = end_time - start_time;
  return t_elapsed.count();
}

void writeMatrixToFile(const Matrix &M, const std::string &filename) {
  std::ofstream file;
  file.open(filename);
  if (!file.is_open()) {
    printf("Cannot write to specified file: %s\n", filename.c_str());
    return;
  }
  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
  file << M.format(CSVFormat);
  file.close();
}

void writeSparseMatrixToFile(const SparseMatrix &M, const std::string &filename) {
  std::ofstream file;
  file.open(filename);
  if (!file.is_open()) {
    printf("Cannot write to specified file: %s\n", filename.c_str());
    return;
  }

  for (int k = 0; k < M.outerSize(); ++k) {
    for (SparseMatrix::InnerIterator it(M, k); it; ++it) {
      file << it.row() << ",";
      file << it.col() << ",";
      file << it.value() << "\n";
    }
  }
  file.close();
}

/**
###############################################################
###############################################################
The following implementations are originally implemented in:

SE-Sync: https://github.com/david-m-rosen/SE-Sync.git

Cartan-Sync: https://bitbucket.org/jesusbriales/cartan-sync/src

###############################################################
###############################################################
*/

std::vector<RelativeSEMeasurement> read_g2o_file(const std::string &filename,
                                                 size_t &num_poses) {
  // Preallocate output vector
  std::vector<RelativeSEMeasurement> measurements;

  // A single measurement, whose values we will fill in
  RelativeSEMeasurement measurement;
  measurement.weight = 1.0;

  // A string used to contain the contents of a single line
  std::string line;

  // A string used to extract tokens from each line one-by-one
  std::string token;

  // Preallocate various useful quantities
  double dx, dy, dz, dtheta, dqx, dqy, dqz, dqw, I11, I12, I13, I14, I15, I16,
      I22, I23, I24, I25, I26, I33, I34, I35, I36, I44, I45, I46, I55, I56, I66;
  size_t i, j;

  // Open the file for reading
  std::ifstream infile(filename);

  // Create Pose ID set
  std::set<size_t> pose_ids;

  while (std::getline(infile, line)) {
    // Construct a stream from the string
    std::stringstream strstrm(line);

    // Extract the first token from the string
    strstrm >> token;

    if (token == "EDGE_SE2") {
      // This is a 2D pose measurement

      /** The g2o format specifies a 2D relative pose measurement in the
       * following form:
       *
       * EDGE_SE2 id1 id2 dx dy dtheta, I11, I12, I13, I22, I23, I33
       *
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dtheta >> I11 >> I12 >> I13 >> I22 >>
              I23 >> I33;

      // Fill in elements of this measurement

      // Pose ids
      measurement.r1 = 0;
      measurement.r2 = 0;
      measurement.p1 = i;
      measurement.p2 = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<double, 2, 1>(dx, dy);
      measurement.R = Eigen::Rotation2Dd(dtheta).toRotationMatrix();

      Eigen::Matrix2d TranCov;
      TranCov << I11, I12, I12, I22;
      measurement.tau = 2 / TranCov.inverse().trace();
      measurement.kappa = I33;

      if (i+1 == j) {
        measurement.fixedWeight = true;
      } else {
        measurement.fixedWeight = false;
      }

    } else if (token == "EDGE_SE3:QUAT") {
      // This is a 3D pose measurement

      /** The g2o format specifies a 3D relative pose measurement in the
       * following form:
       *
       * EDGE_SE3:QUAT id1, id2, dx, dy, dz, dqx, dqy, dqz, dqw
       *
       * I11 I12 I13 I14 I15 I16
       *     I22 I23 I24 I25 I26
       *         I33 I34 I35 I36
       *             I44 I45 I46
       *                 I55 I56
       *                     I66
       */

      // Extract formatted output
      strstrm >> i >> j >> dx >> dy >> dz >> dqx >> dqy >> dqz >> dqw >> I11 >>
              I12 >> I13 >> I14 >> I15 >> I16 >> I22 >> I23 >> I24 >> I25 >> I26 >>
              I33 >> I34 >> I35 >> I36 >> I44 >> I45 >> I46 >> I55 >> I56 >> I66;

      // Fill in elements of the measurement

      // Pose ids
      measurement.r1 = 0;
      measurement.r2 = 0;
      measurement.p1 = i;
      measurement.p2 = j;

      // Raw measurements
      measurement.t = Eigen::Matrix<double, 3, 1>(dx, dy, dz);
      measurement.R = Eigen::Quaterniond(dqw, dqx, dqy, dqz).toRotationMatrix();

      // Compute precisions

      // Compute and store the optimal (information-divergence-minimizing) value
      // of the parameter tau
      Eigen::Matrix3d TranCov;
      TranCov << I11, I12, I13, I12, I22, I23, I13, I23, I33;
      measurement.tau = 3 / TranCov.inverse().trace();

      // Compute and store the optimal (information-divergence-minimizing value
      // of the parameter kappa
      Eigen::Matrix3d RotCov;
      RotCov << I44, I45, I46, I45, I55, I56, I46, I56, I66;
      measurement.kappa = 3 / (2 * RotCov.inverse().trace());

      if (i+1 == j) {
        measurement.fixedWeight = true;
      } else {
        measurement.fixedWeight = false;
      }

    } else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT")) {
      // This is just initialization information, so do nothing
      continue;
    } else if ((token == "FIX")) {
      LOG(WARNING) << "[read_g2o_file] FIX ID_SET is not supported. Skipping line...";
      continue;
    } else {
      LOG(FATAL) << "[read_g2o_file] Unrecognized type: " << token << "!";
    }

    // Update pose IDs
    pose_ids.emplace(measurement.p1);
    pose_ids.emplace(measurement.p2);

    measurements.push_back(measurement);
  }  // while

  infile.close();

  // Get first pose ID
  const size_t first_pose_id = *pose_ids.begin();

  // Check for consecutive sequencing of pose IDs
  size_t prev_pose_id = first_pose_id - 1;
  for (const size_t& pose_id : pose_ids) {
    if (pose_id != prev_pose_id + 1) {
      LOG(FATAL) << "[read_g2o_file] Invalid pose ID sequencing: [" << prev_pose_id << "," << pose_id << "]. "
                 << "The set of pose IDs must be consecutive!";
    }
    prev_pose_id = pose_id;
  }

  // Reindex Pose IDs from zero if necessary
  if (first_pose_id != 0) {
    LOG(WARNING) << "[read_g2o_file] Invalid first pose ID: " << first_pose_id << ". "
                 << "Pose IDs will be re-indexed starting from zero.";

    // Decrement all pose IDs by the first pose ID
    for (RelativeSEMeasurement& measurement : measurements) {
      measurement.p1 -= first_pose_id;
      measurement.p2 -= first_pose_id;
    }
  }

  num_poses = pose_ids.size();

  return measurements;
}

void get_dimension_and_num_poses(const std::vector<RelativeSEMeasurement> &measurements,
                                 size_t &dimension,
                                 size_t &num_poses) {
  CHECK(!measurements.empty());
  dimension = measurements[0].t.size();
  CHECK(dimension == 2 || dimension == 3);
  num_poses = 0;
  for (const auto& meas: measurements) {
    num_poses = std::max(num_poses, meas.p1 + 1);
    num_poses = std::max(num_poses, meas.p2 + 1);
  }
}

void constructOrientedConnectionIncidenceMatrixSE(
    const std::vector<RelativeSEMeasurement> &measurements, SparseMatrix &AT,
    DiagonalMatrix &OmegaT) {
  // Deduce graph dimensions from measurements
  size_t d;  // Dimension of Euclidean space
  d = (!measurements.empty() ? measurements[0].t.size() : 0);
  size_t dh = d + 1;  // Homogenized dimension of Euclidean space
  size_t m;           // Number of measurements
  m = measurements.size();
  size_t n = 0;  // Number of poses
  for (const RelativeSEMeasurement &meas: measurements) {
    if (n < meas.p1) n = meas.p1;
    if (n < meas.p2) n = meas.p2;
  }
  n++;  // Account for 0-based indexing: node indexes go from 0 to max({i,j})

  // Define connection incidence matrix dimensions
  // This is a [n x m] (dh x dh)-block matrix
  size_t rows = (d + 1) * n;
  size_t cols = (d + 1) * m;

  // We use faster ordered insertion, as suggested in
  // https://eigen.tuxfamily.org/dox/group__TutorialSparse.html#TutorialSparseFilling
  Eigen::SparseMatrix<double, Eigen::ColMajor> A(rows, cols);
  A.reserve(Eigen::VectorXi::Constant(cols, 8));
  DiagonalMatrix Omega(cols);  // One block per measurement: (d+1)*m
  DiagonalMatrix::DiagonalVectorType &diagonal = Omega.diagonal();

  // Insert actual measurement values
  size_t i, j;
  for (size_t k = 0; k < m; k++) {
    const RelativeSEMeasurement &meas = measurements[k];
    i = meas.p1;
    j = meas.p2;

    /// Assign SE(d) matrix to block leaving node i
    /// AT(i,k) = -Tij (NOTE: NEGATIVE)
    // Do it column-wise for speed
    // Elements of rotation
    for (size_t c = 0; c < d; c++)
      for (size_t r = 0; r < d; r++)
        A.insert(i * dh + r, k * dh + c) = -meas.R(r, c);

    // Elements of translation
    for (size_t r = 0; r < d; r++)
      A.insert(i * dh + r, k * dh + d) = -meas.t(r);

    // Additional 1 for homogeneization
    A.insert(i * dh + d, k * dh + d) = -1;

    /// Assign (d+1)-identity matrix to block leaving node j
    /// AT(j,k) = +I (NOTE: POSITIVE)
    for (size_t r = 0; r < d + 1; r++) A.insert(j * dh + r, k * dh + r) = +1;

    /// Assign isotropic weights in diagonal matrix
    for (size_t r = 0; r < d; r++) diagonal[k * dh + r] = meas.weight * meas.kappa;

    diagonal[k * dh + d] = meas.weight * meas.tau;
  }

  A.makeCompressed();

  AT = A;
  OmegaT = Omega;
}

SparseMatrix constructConnectionLaplacianSE(
    const std::vector<RelativeSEMeasurement> &measurements) {
  SparseMatrix AT;
  DiagonalMatrix OmegaT;
  constructOrientedConnectionIncidenceMatrixSE(measurements, AT, OmegaT);
  return AT * OmegaT * AT.transpose();
}

void constructBMatrices(const std::vector<RelativeSEMeasurement> &measurements, SparseMatrix &B1,
                        SparseMatrix &B2, SparseMatrix &B3) {
  // Clear input matrices
  B1.setZero();
  B2.setZero();
  B3.setZero();

  size_t num_poses = 0;
  size_t d = (!measurements.empty() ? measurements[0].t.size() : 0);

  std::vector<Eigen::Triplet<double>> triplets;

  // Useful quantities to cache
  size_t d2 = d * d;
  size_t d3 = d * d * d;

  size_t i, j; // Indices for the tail and head of the given measurement
  double sqrttau;
  size_t max_pair;

  /// Construct the matrix B1 from equation (69a) in the tech report
  triplets.reserve(2 * d * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].p1;
    j = measurements[e].p2;
    sqrttau = sqrt(measurements[e].tau);

    // Block corresponding to the tail of the measurement
    for (size_t l = 0; l < d; l++) {
      triplets.emplace_back(e * d + l, i * d + l,
                            -sqrttau); // Diagonal element corresponding to tail
      triplets.emplace_back(e * d + l, j * d + l,
                            sqrttau); // Diagonal element corresponding to head
    }

    // Keep track of the number of poses we've seen
    max_pair = std::max<size_t>(i, j);
    if (max_pair > num_poses)
      num_poses = max_pair;
  }
  num_poses++; // Account for zero-based indexing

  B1.resize(d * measurements.size(), d * num_poses);
  B1.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B2 from equation (69b) in the tech report
  triplets.clear();
  triplets.reserve(d2 * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    i = measurements[e].p1;
    sqrttau = sqrt(measurements[e].tau);
    for (size_t k = 0; k < d; k++)
      for (size_t r = 0; r < d; r++)
        triplets.emplace_back(d * e + r, d2 * i + d * k + r,
                              -sqrttau * measurements[e].t(k));
  }

  B2.resize(d * measurements.size(), d2 * num_poses);
  B2.setFromTriplets(triplets.begin(), triplets.end());

  /// Construct matrix B3 from equation (69c) in the tech report
  triplets.clear();
  triplets.reserve((d3 + d2) * measurements.size());

  for (size_t e = 0; e < measurements.size(); e++) {
    double sqrtkappa = std::sqrt(measurements[e].kappa);
    const Matrix &R = measurements[e].R;

    for (size_t r = 0; r < d; r++)
      for (size_t c = 0; c < d; c++) {
        i = measurements[e].p1; // Tail of measurement
        j = measurements[e].p2; // Head of measurement

        // Representation of the -sqrt(kappa) * Rt(i,j) \otimes I_d block
        for (size_t l = 0; l < d; l++)
          triplets.emplace_back(e * d2 + d * r + l, i * d2 + d * c + l,
                                -sqrtkappa * R(c, r));
      }

    for (size_t l = 0; l < d2; l++)
      triplets.emplace_back(e * d2 + l, j * d2 + l, sqrtkappa);
  }

  B3.resize(d2 * measurements.size(), d2 * num_poses);
  B3.setFromTriplets(triplets.begin(), triplets.end());
}

Matrix recoverTranslations(const SparseMatrix &B1, const SparseMatrix &B2,
                           const Matrix &R) {
  unsigned int d = R.rows();
  unsigned int n = R.cols() / d;

  // Vectorization of R matrix
  Eigen::Map<Eigen::VectorXd> rvec((double *) R.data(), d * d * n);

  // Form the matrix comprised of the right (n-1) block columns of B1
  SparseMatrix B1red = B1.rightCols(d * (n - 1));

  Eigen::VectorXd c = B2 * rvec;

  // Solve
  Eigen::SPQR<SparseMatrix> QR(B1red);
  Eigen::VectorXd tred = -QR.solve(c);

  // Reshape this result into a d x (n-1) matrix
  Eigen::Map<Eigen::MatrixXd> tred_mat(tred.data(), d, n - 1);

  // Allocate output matrix
  Eigen::MatrixXd t = Eigen::MatrixXd::Zero(d, n);

  // Set rightmost n-1 columns
  t.rightCols(n - 1) = tred_mat;

  return t;
}

Matrix projectToRotationGroup(const Matrix &M) {
  // Compute the SVD of M
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

  double detU = svd.matrixU().determinant();
  double detV = svd.matrixV().determinant();

  if (detU * detV > 0) {
    return svd.matrixU() * svd.matrixV().transpose();
  } else {
    Eigen::MatrixXd Uprime = svd.matrixU();
    Uprime.col(Uprime.cols() - 1) *= -1;
    return Uprime * svd.matrixV().transpose();
  }
}

Matrix projectToStiefelManifold(const Matrix &M) {
  size_t r = M.rows();
  size_t d = M.cols();
  CHECK(r >= d);
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU() * svd.matrixV().transpose();
}

Matrix fixedStiefelVariable(unsigned d, unsigned r) {
  std::srand(1);
  ROPTLIB::StieVariable var(r, d);
  var.RandInManifold();
  return Eigen::Map<Matrix>((double *) var.ObtainReadData(), r, d);
}

Matrix randomStiefelVariable(unsigned d, unsigned r) {
  ROPTLIB::StieVariable var(r, d);
  var.RandInManifold();
  return Eigen::Map<Matrix>((double *) var.ObtainReadData(), r, d);
}

double computeMeasurementError(const RelativeSEMeasurement &m,
                               const Matrix &R1, const Matrix &t1,
                               const Matrix &R2, const Matrix &t2) {
  double rotationErrorSq = (R1 * m.R - R2).squaredNorm();
  double translationErrorSq = (t2 - t1 - R1 * m.t).squaredNorm();
  return m.kappa * rotationErrorSq + m.tau * translationErrorSq;
}

double chi2inv(double quantile, size_t dof) {
  boost::math::chi_squared_distribution<double> chi2(dof);
  return boost::math::quantile(chi2, quantile);
}

double angular2ChordalSO3(double rad) {
  return 2 * sqrt(2) * sin(rad / 2);
}

void checkRotationMatrix(const Matrix &R) {
  const auto d = R.rows();
  CHECK(R.cols() == d);
  double err_det = abs(R.determinant() - 1.0);
  double err_norm = (R.transpose() * R - Matrix::Identity(d, d)).norm();
  if (err_det > 1e-5 || err_norm > 1e-5) {
    LOG(WARNING)
        << "[checkRotationMatrix] Invalid rotation: err_det="
        << err_det << ", err_norm=" << err_norm;
  }
}

void checkStiefelMatrix(const Matrix &Y) {
  const auto d = Y.cols();
  double err_norm = (Y.transpose() * Y - Matrix::Identity(d, d)).norm();
  if (err_norm > 1e-5) {
    LOG(WARNING)
        << "[checkStiefelMatrix] Invalid Stiefel: err_norm="
        << err_norm;
  }
}

}  // namespace DPGO
