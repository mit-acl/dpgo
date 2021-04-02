
/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/DPGO_types.h>
#include <DPGO/DPGO_utils.h>
#include <DPGO/PGOAgent.h>
#include <DPGO/QuadraticProblem.h>

#include <cstdlib>
#include <fstream>
#include <iostream>

using namespace std;
using namespace DPGO;

#include <string>
#include <fstream>
#include <vector>
#include <utility> // std::pair
#include <stdexcept> // std::runtime_error
#include <sstream> // std::stringstream

int main(int argc, char** argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 2) {
    cout << "Single robot pose-graph optimization. " << endl;
    cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
    exit(1);
  }

  cout << "Single robot pose-graph optimization demo. " << endl;

  size_t num_poses;
  vector<RelativeSEMeasurement> dataset = read_g2o_file(argv[1], num_poses);

  /**
  ###########################################
  Set parameters for PGOAgent
  ###########################################
  */

  unsigned int n, d, r;    //vector<float> PoseData;
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  n = num_poses;
  r = d;
  PGOAgentParameters options(d, r, 1);
  options.verbose = true;

  vector<RelativeSEMeasurement> odometry;
  vector<RelativeSEMeasurement> private_loop_closures;
  vector<RelativeSEMeasurement> shared_loop_closure;
  for (const auto& mIn : dataset) {
    unsigned srcIdx = mIn.p1;
    unsigned dstIdx = mIn.p2;

    RelativeSEMeasurement m(0, 0, srcIdx, dstIdx, mIn.R, mIn.t,
                            mIn.kappa, mIn.tau);

    if (srcIdx + 1 == dstIdx) {
      // Odometry
      odometry.push_back(m);
    } else {
      // private loop closure
      private_loop_closures.push_back(m);
    }
  }

  // Construct the centralized problem (used for evaluation)
  SparseMatrix QCentral = constructConnectionLaplacianSE(dataset);
  QuadraticProblem problemCentral(n, d, r);
  problemCentral.setQ(QCentral);

  /**
  ###########################################
  Initialization
  ###########################################
  */

  auto* agent = new PGOAgent(0, options);
  agent->setPoseGraph(odometry, private_loop_closures,
                      shared_loop_closure);

  /**
  ###########################################
  Local Pose Graph Optimization
  ###########################################
  */

  cout << "Running local pose graph optimization..." << endl;
  Matrix X = agent->localPoseGraphOptimization();

  // Evaluate
  std::cout << "Cost = " << 2 * problemCentral.f(X) << endl;

  exit(0);
}
