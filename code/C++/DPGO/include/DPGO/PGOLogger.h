/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DISTRIBUTEDPGO_DPGO_INCLUDE_DPGO_PGOLOGGER_H_
#define DISTRIBUTEDPGO_DPGO_INCLUDE_DPGO_PGOLOGGER_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <DPGO/DPGO_types.h>
#include <DPGO/RelativeSEMeasurement.h>

namespace DPGO {

class PGOLogger {
 public:
  /*!
   * @brief Constructor
   * @param logDir directory to store all log files
   */
  explicit PGOLogger(std::string logDir);

  ~PGOLogger();

  /*!
   * @brief log trajectory to file (currently only work for 3D data)
   * @param d dimension (2 or 3)
   * @param n number of poses
   * @param T d-by-(d+1)*n matrix where each d-by-(d+1) block represents a pose
   * @param filename filename
   */
  void logTrajectory(unsigned d, unsigned n, const Matrix &T, const std::string &filename);

  /*!
   * @brief log measurements to file (currently only work for 3D data)
   * @param measurements a vector of relative pose measurements
   * @param filename
   */
  void logMeasurements(std::vector<RelativeSEMeasurement> &measurements, const std::string &filename);

  /**
   * @brief load trajectory from file
   * @param filename
   * @return
   */
  Matrix loadTrajectory(const std::string &filename);

  /**
   * @brief read a list of measurements from file
   * @param filename
   * @param load_weight
   * @return
   */
  std::vector<RelativeSEMeasurement> loadMeasurements(const std::string &filename, bool load_weight = false);

 private:
  std::string logDirectory;
};

} // end namespace DPGO

#endif //DISTRIBUTEDPGO_DPGO_INCLUDE_DPGO_PGOLOGGER_H_
