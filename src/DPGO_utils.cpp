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
#include <cassert>
#include <boost/math/distributions/chi_squared.hpp>

namespace DPGO {

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
  std::vector<DPGO::RelativeSEMeasurement> measurements;

  // A single measurement, whose values we will fill in
  DPGO::RelativeSEMeasurement measurement;
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

  num_poses = 0;

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

    } else if ((token == "VERTEX_SE2") || (token == "VERTEX_SE3:QUAT")) {
      // This is just initialization information, so do nothing
      continue;
    } else {
      std::cout << "Error: unrecognized type: " << token << "!" << std::endl;
      assert(false);
    }

    // Update maximum value of poses found so far
    size_t max_pair = std::max<double>(measurement.p1, measurement.p2);

    num_poses = ((max_pair > num_poses) ? max_pair : num_poses);
    measurements.push_back(measurement);
  }  // while

  infile.close();

  num_poses++;  // Account for the use of zero-based indexing

  return measurements;
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

Matrix chordalInitialization(
    size_t dimension, size_t num_poses,
    const std::vector<RelativeSEMeasurement> &measurements) {
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3);

  // Recover rotations
  size_t d = (!measurements.empty() ? measurements[0].t.size() : 0);
  unsigned int d2 = d * d;
  assert(dimension == d);
  assert(num_poses == (unsigned) B3.cols() / d2);

  SparseMatrix B3red = B3.rightCols((num_poses - 1) * d2);
  B3red.makeCompressed();  // Must be in compressed format to use
  // Eigen::SparseQR!

  // Vectorization of I_d
  Eigen::MatrixXd Id = Eigen::MatrixXd::Identity(d, d);
  Eigen::Map<Eigen::VectorXd> Id_vec(Id.data(), d2);

  Eigen::VectorXd cR = B3.leftCols(d2) * Id_vec;

  Eigen::VectorXd rvec;
  Eigen::SPQR<SparseMatrix> QR(B3red);
  rvec = -QR.solve(cR);

  Matrix Rchordal(d, d * num_poses);
  Rchordal.leftCols(d) = Id;
  Rchordal.rightCols((num_poses - 1) * d) =
      Eigen::Map<Eigen::MatrixXd>(rvec.data(), d, (num_poses - 1) * d);
  for (unsigned int i = 1; i < num_poses; i++)
    Rchordal.block(0, i * d, d, d) =
        projectToRotationGroup(Rchordal.block(0, i * d, d, d));

  // Recover translation
  Matrix tchordal = recoverTranslations(B1, B2, Rchordal);
  assert((unsigned) tchordal.rows() == dimension);
  assert((unsigned) tchordal.cols() == num_poses);

  // Assemble full pose
  Matrix Tchordal(d, num_poses * (d + 1));
  for (size_t i = 0; i < num_poses; i++) {
    Tchordal.block(0, i * (d + 1), d, d) = Rchordal.block(0, i * d, d, d);
    Tchordal.block(0, i * (d + 1) + d, d, 1) = tchordal.block(0, i, d, 1);
  }

  return Tchordal;
}

Matrix odometryInitialization(size_t dimension, size_t num_poses, const std::vector<RelativeSEMeasurement> &odometry) {
  size_t d = dimension;
  size_t n = num_poses;

  Matrix T(d, n * (d + 1));
  // Initialize first pose to be identity
  T.block(0, 0, d, d) = Matrix::Identity(d, d);
  T.block(0, d, d, 1) = Matrix::Zero(d, 1);
  for (size_t src = 0; src < odometry.size(); ++src) {
    size_t dst = src + 1;
    const RelativeSEMeasurement &m = odometry[src];
    assert(m.p1 == src);
    assert(m.p2 == dst);
    Matrix Rsrc = T.block(0, src * (d + 1), d, d);
    Matrix tsrc = T.block(0, src * (d + 1) + d, d, 1);
    Matrix Rdst = Rsrc * m.R;
    Matrix tdst = tsrc + Rsrc * m.t;
    T.block(0, dst * (d + 1), d, d) = Rdst;
    T.block(0, dst * (d + 1) + d, d, 1) = tdst;
  }
  return T;
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
  assert(r >= d);
  Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  return svd.matrixU() * svd.matrixV().transpose();
}

Matrix fixedStiefelVariable(unsigned d, unsigned r) {
  std::srand(1);
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
  assert(R.cols() == d);
  assert(abs(R.determinant() - 1.0) < 1e-8);
  assert((R.transpose() * R - Matrix::Identity(d, d)).norm() < 1e-8);
}

void singleTranslationAveraging(Vector &tOpt,
                                const std::vector<Vector> &tVec,
                                const Vector &tau) {
  const int n = (int) tVec.size();
  assert(n > 0);
  const auto d = tVec[0].rows();
  Vector tau_ = Vector::Ones(n);
  if (tau.rows() == n) {
    tau_ = tau;
  }
  Vector s = Vector::Zero(d);
  double w = 0;
  for (Eigen::Index i = 0; i < n; ++i) {
    s += tau_(i) * tVec[i];
    w += tau_(i);
  }
  tOpt = s / w;
}

void singleRotationAveraging(Matrix &ROpt,
                             const std::vector<Matrix> &RVec,
                             const Vector &kappa) {
  const int n = (int) RVec.size();
  assert(n > 0);
  const auto d = RVec[0].rows();
  Vector kappa_ = Vector::Ones(n);
  if (kappa.rows() == n) {
    kappa_ = kappa;
  }
  Matrix M = Matrix::Zero(d, d);
  for (Eigen::Index i = 0; i < n; ++i) {
    M += kappa_(i) * RVec[i];
  }
  ROpt = projectToRotationGroup(M);
}

void singlePoseAveraging(Matrix &ROpt, Vector &tOpt,
                         const std::vector<Matrix> &RVec,
                         const std::vector<Vector> &tVec,
                         const Vector &kappa,
                         const Vector &tau) {
  assert(!RVec.empty());
  assert(!tVec.empty());
  assert(RVec.size() == tVec.size());
  assert(RVec[0].rows() == tVec[0].rows());
  singleTranslationAveraging(tOpt, tVec, tau);
  singleRotationAveraging(ROpt, RVec, kappa);
}

void robustSingleRotationAveraging(Matrix &ROpt,
                                   std::vector<size_t> &inlierIndices,
                                   const std::vector<Matrix> &RVec,
                                   const Vector &kappa,
                                   double errorThreshold) {
  const double w_tol = 1e-8;
  const int n = (int) RVec.size();
  assert(n > 0);
  Vector kappa_ = Vector::Ones(n);
  Vector weights_ = Vector::Ones(n);
  if (kappa.rows() == n) {
    kappa_ = kappa;
  }
  for (const auto &Ri: RVec) {
    checkRotationMatrix(Ri);
  }
  // Initialize estimate
  singleRotationAveraging(ROpt, RVec, kappa_);
  Vector rSqVec = Vector::Zero(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    rSqVec(i) = kappa_(i) * (ROpt - RVec[i]).squaredNorm();
  }
  // Initialize robust cost
  double barc = errorThreshold;
  double barcSq = barc * barc;
  double muInit = barcSq / (2 * rSqVec.maxCoeff() - barcSq);
  muInit = std::min(muInit, 1e-5);
  // Negative values of initial mu corresponds to small residual errors. In this case skip applying GNC.
  if (muInit > 0) {
    RobustCostParameters params;
    params.GNCBarc = barc;
    params.GNCMaxNumIters = 1000;
    params.GNCInitMu = muInit;
    RobustCost cost(RobustCostType::GNC_TLS, params);
    for (unsigned iter = 0; iter < params.GNCMaxNumIters; ++iter) {
      // Update solution
      singleRotationAveraging(ROpt, RVec, kappa_.cwiseProduct(weights_));
      // Update weight
      int nc = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        double rSq = kappa_(i) * (ROpt - RVec[i]).squaredNorm();
        double wi = cost.weight(sqrt(rSq));
        if (wi < w_tol || wi > 1 - w_tol) {
          nc++;
        }
        weights_(i) = wi;
      }
      if (nc == n) {
        break;
      }
      // Update GNC
      cost.update();
    }
  }
  // Retrieve inliers
  inlierIndices.clear();
  for (Eigen::Index i = 0; i < n; ++i) {
    double wi = weights_(i);
    if (wi > 1 - w_tol) {
      inlierIndices.push_back(i);
    }
  }
}

void robustSinglePoseAveraging(Matrix &ROpt, Vector &tOpt,
                               std::vector<size_t> &inlierIndices,
                               const std::vector<Matrix> &RVec,
                               const std::vector<Vector> &tVec,
                               const Vector &kappa,
                               const Vector &tau,
                               double errorThreshold) {
  const double w_tol = 1e-8;
  const int n = (int) RVec.size();
  assert(n > 0);
  assert(tVec.size() == n);
  Vector kappa_ = 10000 * Vector::Ones(n);
  Vector tau_ = 100 * Vector::Ones(n);
  Vector weights_ = Vector::Ones(n);
  if (kappa.rows() == n) {
    kappa_ = kappa;
  }
  if (tau.rows() == n) {
    tau_ = tau;
  }
  for (const auto &Ri: RVec) {
    checkRotationMatrix(Ri);
  }
  // Initialize estimate
  singlePoseAveraging(ROpt,
                      tOpt,
                      RVec,
                      tVec,
                      kappa_.cwiseProduct(weights_),
                      tau_.cwiseProduct(weights_));
  Vector rSqVec = Vector::Zero(n);
  for (Eigen::Index i = 0; i < n; ++i) {
    rSqVec(i) = kappa_(i) * (ROpt - RVec[i]).squaredNorm() + tau_(i) * (tOpt - tVec[i]).squaredNorm();
  }
  // Initialize robust cost
  double barc = errorThreshold;
  double barcSq = barc * barc;
  double muInit = barcSq / (2 * rSqVec.maxCoeff() - barcSq);
  muInit = std::min(muInit, 1e-5);
  // Negative values of initial mu corresponds to small residual errors. In this case skip applying GNC.
  if (muInit > 0) {
    RobustCostParameters params;
    params.GNCBarc = barc;
    params.GNCMaxNumIters = 10000;
    params.GNCInitMu = muInit;
    RobustCost cost(RobustCostType::GNC_TLS, params);
    unsigned iter = 0;
    for (iter = 0; iter < params.GNCMaxNumIters; ++iter) {
      // Update solution
      singlePoseAveraging(ROpt,
                          tOpt,
                          RVec,
                          tVec,
                          kappa_.cwiseProduct(weights_),
                          tau_.cwiseProduct(weights_));
      // Update weight
      int nc = 0;
      for (Eigen::Index i = 0; i < n; ++i) {
        double rSq = kappa_(i) * (ROpt - RVec[i]).squaredNorm() + tau_(i) * (tOpt - tVec[i]).squaredNorm();
        double wi = cost.weight(sqrt(rSq));
        if (wi < w_tol || wi > 1 - w_tol) {
          nc++;
        }
        weights_(i) = wi;
      }
      if (nc == n) {
        break;
      }
      // Update GNC
      cost.update();
    }
  }
  // Retrieve inliers
  inlierIndices.clear();
  for (Eigen::Index i = 0; i < n; ++i) {
    double wi = weights_(i);
    if (wi > 1 - w_tol) {
      inlierIndices.push_back(i);
    }
  }
}

}  // namespace DPGO
