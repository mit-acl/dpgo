#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_solver.h>
#include <DPGO/DPGO_robust.h>
#include <DPGO/PoseGraph.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/QuadraticOptimizer.h>
#include <iostream>
#include <random>

#include "gtest/gtest.h"

using namespace DPGO;

TEST(testDPGO, testRobustSingleRotationAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const double cbar = angular2ChordalSO3(0.5);  // approximately 30 deg
    std::vector<Matrix> RVec;
    RVec.push_back(RTrue);
    Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = Vector::Ones(1);
    robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa, cbar);
    checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDPGO, testRobustSingleRotationAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double tol = angular2ChordalSO3(0.02);
    const double cbar = angular2ChordalSO3(0.3);
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    std::vector<Matrix> RVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      if ((RRand - RTrue).norm() > 1.2 * cbar)  // Make sure that outlier is separated from the true rotation
        RVec.emplace_back(RRand);
    }
    Matrix ROpt;
    std::vector<size_t> inlierIndices;
    const auto kappa = Vector::Ones(50);
    robustSingleRotationAveraging(ROpt, inlierIndices, RVec, kappa, cbar);
    checkRotationMatrix(ROpt);
    double distChordal = (ROpt - RTrue).norm();
    ASSERT_LE(distChordal, tol);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}

TEST(testDPGO, testRobustSinglePoseAveragingTrivial) {
  for (int trial = 0; trial < 50; ++trial) {
    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<Matrix> RVec;
    RVec.push_back(RTrue);
    std::vector<Vector> tVec;
    tVec.push_back(tTrue);
    const auto kappa = 10000 * Vector::Ones(1);
    const auto tau = 100 * Vector::Ones(1);
    const double gnc_quantile = 0.9;
    const double gnc_barc = RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    Matrix ROpt;
    Vector tOpt;
    std::vector<size_t> inlierIndices;
    robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa, tau, gnc_barc);
    checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), 1e-8);
    ASSERT_LE((tOpt - tTrue).norm(), 1e-8);
    ASSERT_EQ(inlierIndices.size(), 1);
    ASSERT_EQ(inlierIndices[0], 0);
  }
}

TEST(testDPGO, testRobustSinglePoseAveraging) {
  for (int trial = 0; trial < 50; ++trial) {
    const double RMaxError = angular2ChordalSO3(0.02);
    const double tMaxError = 1e-2;
    const double gnc_quantile = 0.9;
    const double gnc_barc = RobustCost::computeErrorThresholdAtQuantile(gnc_quantile, 3);
    const double kappa = 10000;
    const double tau = 100;
    const auto kappa_vec = kappa * Vector::Ones(50);
    const auto tau_vec = tau * Vector::Ones(50);

    const Matrix RTrue = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    const Vector tTrue = Eigen::Vector3d::Zero();
    std::vector<Matrix> RVec;
    std::vector<Vector> tVec;
    // Push inliers
    for (int i = 0; i < 10; ++i) {
      RVec.emplace_back(RTrue);
      tVec.emplace_back(tTrue);
    }
    // Push outliers
    while (RVec.size() < 50) {
      Matrix RRand = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
      Matrix tRand = Eigen::Vector3d::Random();
      double rSq = kappa * (RTrue - RRand).squaredNorm() + tau * (tTrue - tRand).squaredNorm();
      if (std::sqrt(rSq) > 1.2 * gnc_barc) { // Make sure that outliers are sufficiently far away from ground truth
        RVec.emplace_back(RRand);
        tVec.emplace_back(tRand);
      }
    }
    Matrix ROpt;
    Vector tOpt;
    std::vector<size_t> inlierIndices;
    robustSinglePoseAveraging(ROpt, tOpt, inlierIndices, RVec, tVec, kappa_vec, tau_vec, gnc_barc);
    checkRotationMatrix(ROpt);
    ASSERT_LE((ROpt - RTrue).norm(), RMaxError);
    ASSERT_LE((tOpt - tTrue).norm(), tMaxError);
    ASSERT_EQ(inlierIndices.size(), 10);
    for (int i = 0; i < 10; ++i) {
      ASSERT_EQ(inlierIndices[i], i);
    }
  }
}


TEST(testDPGO, testPrior) {
  size_t dimension = 3;
  size_t num_poses = 2;
  size_t robot_id = 0;
  
  // Odometry measurement
  RelativeSEMeasurement m;
  m.r1 = 0;
  m.p1 = 0;
  m.r2 = 0;
  m.p2 = 1;
  m.R = Eigen::Matrix3d::Identity();
  m.t = Eigen::Vector3d::Zero();
  m.kappa = 10000;
  m.tau = 100;
  m.weight = 1;
  m.fixedWeight = true;
  std::vector<RelativeSEMeasurement> measurements;
  measurements.push_back(m);

  PoseArray T(dimension, num_poses);
  T = odometryInitialization(measurements);

  // Form pose graph and add a prior
  auto pose_graph = std::make_shared<PoseGraph>(robot_id, dimension, dimension);
  pose_graph->setMeasurements(measurements);
  Matrix prior_rotation(dimension, dimension);
  prior_rotation << 0.7236,    0.1817,    0.6658,
              -0.6100,    0.6198,    0.4938,
              -0.3230,   -0.7634,    0.5594;
  prior_rotation = projectToRotationGroup(prior_rotation);
  Pose prior(dimension);
  prior.rotation() = prior_rotation;
  pose_graph->setPrior(1, prior);
  QuadraticProblem problem(pose_graph);

  // The odometry initial guess does not respect the prior
  double error0 = (T.pose(0) - prior.pose()).norm();
  double error1 = (T.pose(1) - prior.pose()).norm();
  ASSERT_GT(error0, 1e-6);
  ASSERT_GT(error1, 1e-6);

  // Initialize optimizer object
  ROptParameters params;
  params.verbose = false;
  params.RTR_iterations = 50;
  params.RTR_tCG_iterations = 500;
  params.gradnorm_tol = 1e-5;
  QuadraticOptimizer optimizer(&problem, params);

  // Optimize!
  auto Topt_mat = optimizer.optimize(T.getData());
  T.setData(Topt_mat);
  
  // After optimization, the solution should be fixed on the prior
  error0 = (T.pose(0) - prior.pose()).norm();
  error1 = (T.pose(1) - prior.pose()).norm();
  ASSERT_LT(error0, 1e-6);
  ASSERT_LT(error1, 1e-6);
}


TEST(testDPGO, testRobustPGO) {
  int d = 3;
  int n = 4;
  double kappa = 10000;
  double tau = 100;
  std::vector<Pose> poses_gt;
  for (int i = 0; i < n; ++i) {
    Pose Ti(d);
    Ti.rotation() = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
    Ti.translation() = i * Eigen::Vector3d::Ones();
    poses_gt.push_back(Ti);
  }
  std::vector<RelativeSEMeasurement> measurements;
  // generate odometry
  for (int i = 0; i < n-1; ++i) {
    int j = i + 1;
    Pose Ti = poses_gt[i];
    Pose Tj = poses_gt[j];
    Pose Tij = Ti.inverse() * Tj;
    RelativeSEMeasurement m;
    m.r1 = 0;
    m.r2 = 0;
    m.p1 = i;
    m.p2 = j;
    m.kappa = kappa;
    m.tau = tau;
    m.fixedWeight = true;
    m.R = Tij.rotation();
    m.t = Tij.translation();
    measurements.push_back(m);
  }
  // generate a single inlier loop closure
  Pose Ti = poses_gt[0];
  Pose Tj = poses_gt[3];
  Pose Tij = Ti.inverse() * Tj;
  RelativeSEMeasurement m_inlier;
  m_inlier.r1 = 0;
  m_inlier.r2 = 0;
  m_inlier.p1 = 0;
  m_inlier.p2 = 3;
  m_inlier.kappa = kappa;
  m_inlier.tau = tau;
  m_inlier.fixedWeight = false;
  m_inlier.R = Tij.rotation();
  m_inlier.t = Tij.translation();
  measurements.push_back(m_inlier);
  // generate a single outlier loop closure
  RelativeSEMeasurement m_outlier;
  m_outlier.r1 = 0;
  m_outlier.r2 = 0;
  m_outlier.p1 = 1;
  m_outlier.p2 = 3;
  m_outlier.kappa = kappa;
  m_outlier.tau = tau;
  m_outlier.fixedWeight = false;
  m_outlier.R = Eigen::Quaterniond::UnitRandom().toRotationMatrix();
  m_outlier.t = Eigen::Vector3d::Zero();
  measurements.push_back(m_outlier);
  // Solve!
  auto pose_graph = std::make_shared<PoseGraph>(0, d, d);
  pose_graph->setMeasurements(measurements);
  solveRobustPGOParams params;
  params.verbose = false;
  params.opt_params.verbose = false;
  params.opt_params.gradnorm_tol = 1e-1;
  params.opt_params.RTR_iterations = 50;
  params.robust_params.GNCBarc = 7.0;
  PoseArray TOdom = odometryInitialization(pose_graph->odometry());
  auto mutable_measurements = measurements;
  PoseArray T = solveRobustPGO(mutable_measurements, params, &TOdom);
  // Check classification of inlier vs outlier
  for (const auto& m: mutable_measurements) {
    if (!m.fixedWeight) {
      if (m.p1 == 0 && m.p2 == 3)
        CHECK_NEAR(m.weight, 1, 1e-6);
      if (m.p1 == 1 && m.p2 == 3)
        CHECK_NEAR(m.weight, 0, 1e-6);
    }
  }
}