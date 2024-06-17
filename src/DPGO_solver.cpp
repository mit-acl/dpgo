/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_solver.h>
#include <DPGO/DPGO_robust.h>
#include <DPGO/PoseGraph.h>
#include <DPGO/QuadraticOptimizer.h>
#include <Eigen/Geometry>
#include <Eigen/SPQRSupport>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <boost/math/distributions/chi_squared.hpp>
#include <glog/logging.h>

namespace DPGO {

void singleTranslationAveraging(Vector &tOpt,
                                const std::vector<Vector> &tVec,
                                const Vector &tau) {
  const int n = (int) tVec.size();
  CHECK(n > 0);
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
  CHECK(n > 0);
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
  CHECK(!RVec.empty());
  CHECK(!tVec.empty());
  CHECK(RVec.size() == tVec.size());
  CHECK(RVec[0].rows() == tVec[0].rows());
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
  CHECK(n > 0);
  Vector kappa_ = Vector::Ones(n);
  Vector weights_ = Vector::Ones(n);
  if (kappa.rows() == n) {
    kappa_ = kappa;
  }
  for (const auto &Ri : RVec) {
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
    params.costType = RobustCostParameters::Type::GNC_TLS;
    params.GNCBarc = barc;
    params.GNCMaxNumIters = 1000;
    params.GNCInitMu = muInit;
    RobustCost cost(params);
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
  CHECK(n > 0);
  CHECK(tVec.size() == RVec.size());
  Vector kappa_ = 10000 * Vector::Ones(n);
  Vector tau_ = 100 * Vector::Ones(n);
  Vector weights_ = Vector::Ones(n);
  if (kappa.rows() == n) {
    kappa_ = kappa;
  }
  if (tau.rows() == n) {
    tau_ = tau;
  }
  for (const auto &Ri : RVec) {
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
    params.costType = RobustCostParameters::Type::GNC_TLS;
    params.GNCBarc = barc;
    params.GNCMaxNumIters = 10000;
    params.GNCInitMu = muInit;
    RobustCost cost(params);
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

PoseArray chordalInitialization(const std::vector<RelativeSEMeasurement> &measurements) {
  size_t dimension, num_poses;
  get_dimension_and_num_poses(measurements, dimension, num_poses);
  SparseMatrix B1, B2, B3;
  constructBMatrices(measurements, B1, B2, B3);

  // Recover rotations
  size_t d = (!measurements.empty() ? measurements[0].t.size() : 0);
  unsigned int d2 = d * d;
  CHECK(dimension == d);
  CHECK(num_poses == (unsigned) B3.cols() / d2);

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
  CHECK((unsigned) tchordal.rows() == dimension);
  CHECK((unsigned) tchordal.cols() == num_poses);

  // Assemble full pose
  Matrix Tchordal(d, num_poses * (d + 1));
  for (size_t i = 0; i < num_poses; i++) {
    Tchordal.block(0, i * (d + 1), d, d) = Rchordal.block(0, i * d, d, d);
    Tchordal.block(0, i * (d + 1) + d, d, 1) = tchordal.block(0, i, d, 1);
  }

  PoseArray output(dimension, num_poses);
  output.setData(Tchordal);
  return output;
}

PoseArray odometryInitialization(
    const std::vector<RelativeSEMeasurement> &odometry,
    const PoseArray *partial_trajectory) {
  size_t dimension, num_poses;
  get_dimension_and_num_poses(odometry, dimension, num_poses);

  PoseArray T(dimension, num_poses);
  unsigned int next_index = 0;  // Index of next pose to initialize
  // Use partial trajectory if provided
  if (partial_trajectory && partial_trajectory->n() > 0) {
    CHECK_EQ(partial_trajectory->d(), dimension);
    while (next_index < partial_trajectory->n() && next_index < num_poses) {
      T.pose(next_index) = partial_trajectory->pose(next_index);
      next_index++;
    }
  } else {
    T.pose(0) = Pose::Identity(dimension).getData();
    next_index = 1;
  }
  // Initialize the remaining poses using odometry
  for (size_t dst = next_index; dst < num_poses; ++dst) {
    size_t src = dst - 1;
    const RelativeSEMeasurement &m = odometry[src];
    CHECK(m.p1 == src) << ": [odometryInitialization] Error: Odometry measurements are discontinuous at pose: " << src;
    CHECK(m.p2 == dst) << ": [odometryInitialization] Error: Odometry measurements are discontinuous at pose: " << dst;
    const Matrix Rsrc = T.rotation(src);
    const Matrix tsrc = T.translation(src);
    T.rotation(dst) = Rsrc * m.R;
    T.translation(dst) = tsrc + Rsrc * m.t;
  }

  return T;
}

PoseArray solvePGO(const std::vector<RelativeSEMeasurement> &measurements,
                   const ROptParameters &params,
                   const PoseArray *T0) {

  size_t dimension, num_poses, robot_id;
  get_dimension_and_num_poses(measurements, dimension, num_poses);
  robot_id = measurements[0].r1;
  PoseArray T(dimension, num_poses);
  if (T0) {
    T = *T0;
  } else {
    T = chordalInitialization(measurements);
  }
  CHECK_EQ(T.d(), dimension);
  CHECK_EQ(T.n(), num_poses);

  // Form optimization problem
  auto pose_graph = std::make_shared<PoseGraph>(robot_id, dimension, dimension);
  pose_graph->setMeasurements(measurements);
  QuadraticProblem problem(pose_graph);

  // Initialize optimizer object
  QuadraticOptimizer optimizer(&problem, params);

  // Optimize
  auto Topt_mat = optimizer.optimize(T.getData());
  T.setData(Topt_mat);
  return T;
}

PoseArray solveRobustPGO(std::vector<RelativeSEMeasurement> &mutable_measurements,
                         const solveRobustPGOParams &params,
                         const PoseArray *T0) {
  size_t dimension, num_poses;
  get_dimension_and_num_poses(mutable_measurements, dimension, num_poses);
  const double w_tol = 1e-8;
  const int m = (int) mutable_measurements.size();
  // Initialize estimate
  PoseArray T = solvePGO(mutable_measurements, params.opt_params, T0);
  Vector rSqVec = Vector::Zero(m);
  for (int i = 0; i < m; ++i) {
    RelativeSEMeasurement &meas = mutable_measurements[i];
    meas.weight = 1.0;
    rSqVec(i) = computeMeasurementError(meas,
                                        T.rotation(meas.p1),
                                        T.translation(meas.p1),
                                        T.rotation(meas.p2),
                                        T.translation(meas.p2));
  }
  // Initialize robust cost
  CHECK(params.robust_params.costType == RobustCostParameters::Type::GNC_TLS);
  double barc = params.robust_params.GNCBarc;
  double barcSq = barc * barc;
  double muInit = barcSq / (2 * rSqVec.maxCoeff() - barcSq);

  RobustCostParameters params_gnc;
  params_gnc = params.robust_params;
  params_gnc.GNCInitMu = muInit;
  // muInit = std::min(muInit, 1e-5);
  if (params.verbose)
    LOG(INFO) << "[solveRobustPGO] Initial value for mu: " << muInit;
  // Negative values of initial mu corresponds to small residual errors. In this case skip applying GNC.
  if (muInit > 0) {
    RobustCost cost(params_gnc);
    unsigned iter = 0;
    for (iter = 0; iter < params_gnc.GNCMaxNumIters; ++iter) {
      // Update solution
      T = solvePGO(mutable_measurements, params.opt_params, T0);
      // Update weight
      for (int i = 0; i < m; ++i) {
        RelativeSEMeasurement &meas = mutable_measurements[i];
        if (meas.fixedWeight) continue;
        double rSq = computeMeasurementError(meas,
                                             T.rotation(meas.p1),
                                             T.translation(meas.p1),
                                             T.rotation(meas.p2),
                                             T.translation(meas.p2));
        meas.weight = cost.weight(sqrt(rSq));
        // LOG(INFO) << "Residual:" << sqrt(rSq) << ", weight=" << meas.weight;
      }
      // Compute stats
      int num_inliers = 0;
      int num_outliers = 0;
      int num_undecided = 0;
      for (const auto& meas: mutable_measurements) {
        if (meas.fixedWeight) continue;
        if (meas.weight < w_tol) {
          num_outliers++;
        } else if (meas.weight > 1.0 - w_tol) {
          num_inliers++;
        } else {
          num_undecided++;
        }
      }
      if (params.verbose) {
        LOG(INFO) << "[solveRobustPGO] Iteration " << iter << ": "
                  << num_inliers << " inliers, " << num_outliers << " outliers, " << num_undecided << " undecided.";
      }
      if (num_undecided == 0) {
        break;
      }
      // Update GNC
      cost.update();
    }
  }
  T = solvePGO(mutable_measurements, params.opt_params, T0);
  return T;
}

}
