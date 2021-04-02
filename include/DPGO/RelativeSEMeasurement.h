/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef RELATIVESEMEASUREMENT_H
#define RELATIVESEMEASUREMENT_H

#include <DPGO/DPGO_types.h>

#include <Eigen/Dense>
#include <iostream>

namespace DPGO {

/** A simple struct that contains the elements of a relative SE measurement
    from pose (r1, p1) to (r2, p2)
 */
struct RelativeSEMeasurement {
  /** 0-based index of first robot */
  size_t r1;

  /** 0-based index of second robot */
  size_t r2;

  /** 0-based index of first pose */
  size_t p1;

  /** 0-based index of second pose */
  size_t p2;

  /** Rotational measurement */
  Matrix R;

  /** Translational measurement */
  Matrix t;

  /** Rotational measurement precision */
  double kappa;

  /** Translational measurement precision */
  double tau;

  /** If this measurement is an known inlier */
  bool isKnownInlier;

  /** Weight between (0,1) used in Graduated Non-Convexity */
  double weight;

  /** Simple default constructor; does nothing */
  RelativeSEMeasurement() = default;

  /** Basic constructor */
  RelativeSEMeasurement(size_t first_robot, size_t second_robot,
                        size_t first_pose, size_t second_pose,
                        const Eigen::MatrixXd &relative_rotation,
                        const Eigen::VectorXd &relative_translation,
                        double rotational_precision,
                        double translational_precision)
      : r1(first_robot),
        r2(second_robot),
        p1(first_pose),
        p2(second_pose),
        R(relative_rotation),
        t(relative_translation),
        kappa(rotational_precision),
        tau(translational_precision),
        isKnownInlier(false),
        weight(1.0) {}

  /** A utility function for streaming this struct to cout */
  inline friend std::ostream &operator<<(
      std::ostream &os, const RelativeSEMeasurement &measurement) {
    os << "r1: " << measurement.r1 << std::endl;
    os << "p1: " << measurement.p1 << std::endl;
    os << "r2: " << measurement.r2 << std::endl;
    os << "p2: " << measurement.p2 << std::endl;
    os << "R: " << std::endl << measurement.R << std::endl;
    os << "t: " << std::endl << measurement.t << std::endl;
    os << "Kappa: " << measurement.kappa << std::endl;
    os << "Tau: " << measurement.tau << std::endl;
    os << "Is known inlier: " << measurement.isKnownInlier << std::endl;
    os << "Weight: " << measurement.weight << std::endl;

    return os;
  }
};
};  // namespace DPGO
#endif
