
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
#include <cassert>
#include <fstream>
#include <iostream>

using namespace std;
using namespace DPGO;

int main(int argc, char **argv) {
  /**
  ###########################################
  Parse input dataset
  ###########################################
  */

  if (argc < 3) {
    cout << "Distributed pose-graph optimization. " << endl;
    cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]" << endl;
    exit(1);
  }

  cout << "Distributed pose-graph optimization demo. " << endl;

  int num_robots = atoi(argv[1]);
  if (num_robots <= 0) {
    cout << "Number of robots must be positive!" << endl;
    exit(1);
  }
  cout << "Simulating " << num_robots << " robots." << endl;

  size_t num_poses;
  vector<RelativeSEMeasurement> dataset = read_g2o_file(argv[2], num_poses);
  cout << "Loaded dataset from file " << argv[2] << "." << endl;

  /**
  ###########################################
  Set parameters for PGOAgent
  ###########################################
  */

  unsigned int n, d, r;
  SparseMatrix ConLapT = constructConnectionLaplacianSE(dataset);
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  n = num_poses;
  r = 5;
  PGOAgentParameters options(d, r, num_robots);

  /**
  ###########################################
  Prepare dataset
  ###########################################
  */
  unsigned int num_poses_per_robot = num_poses / num_robots;
  if (num_poses_per_robot <= 0) {
    cout << "More robots than total number of poses! Decrease the number of "
            "robots"
         << endl;
    exit(1);
  }

  // create mapping from global pose index to local pose index
  map<unsigned, PoseID> PoseMap;
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    unsigned startIdx = robot * num_poses_per_robot;
    unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
    if (robot == (unsigned) num_robots - 1) endIdx = n;
    for (unsigned idx = startIdx; idx < endIdx; ++idx) {
      unsigned localIdx = idx - startIdx;  // this is the local ID of this pose
      PoseID pose = make_pair(robot, localIdx);
      PoseMap[idx] = pose;
    }
  }

  vector<vector<RelativeSEMeasurement>> odometry(num_robots);
  vector<vector<RelativeSEMeasurement>> private_loop_closures(num_robots);
  vector<vector<RelativeSEMeasurement>> shared_loop_closure(num_robots);
  for (size_t k = 0; k < dataset.size(); ++k) {
    RelativeSEMeasurement mIn = dataset[k];
    PoseID src = PoseMap[mIn.p1];
    PoseID dst = PoseMap[mIn.p2];

    unsigned srcRobot = src.first;
    unsigned srcIdx = src.second;
    unsigned dstRobot = dst.first;
    unsigned dstIdx = dst.second;

    RelativeSEMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx, mIn.R, mIn.t,
                            mIn.kappa, mIn.tau);

    if (srcRobot == dstRobot) {
      // private measurement
      if (srcIdx + 1 == dstIdx) {
        // Odometry
        odometry[srcRobot].push_back(m);
      } else {
        // private loop closure
        private_loop_closures[srcRobot].push_back(m);
      }
    } else {
      // shared measurement
      shared_loop_closure[srcRobot].push_back(m);
      shared_loop_closure[dstRobot].push_back(m);
    }
  }

  /**
  ###########################################
  Initialization
  ###########################################
  */
  vector<PGOAgent *> agents;
  for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
    PGOAgent *agent = new PGOAgent(robot, options);

    // All agents share a special, common matrix called the 'lifting matrix' which the first agent will generate
    if (robot > 0) {
      Matrix M;
      agents[0]->getLiftingMatrix(M);
      agent->setLiftingMatrix(M);
    }

    agent->setPoseGraph(odometry[robot], private_loop_closures[robot],
                        shared_loop_closure[robot]);
    agents.push_back(agent);
  }

  /**
  ###########################################
  Optimization loop
  ###########################################
  */

  unsigned numIters = 500;
  Matrix Xopt(r, n * (d + 1));
  cout << "Running RBCD for " << numIters << " iterations..." << endl;
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, num_robots - 1);

  for (unsigned iter = 0; iter < numIters; ++iter) {
    // Exchange public poses
    for (unsigned robot1 = 0; robot1 < (unsigned) num_robots; ++robot1) {
      PoseDict sharedPoses;
      if (!agents[robot1]->getSharedPoseDict(sharedPoses)) {
        continue;
      }
      for (unsigned robot2 = 0; robot2 < (unsigned) num_robots; ++robot2) {
        if (robot1 == robot2) continue;

        for (auto it = sharedPoses.begin(); it != sharedPoses.end(); ++it) {
          PoseID nID = it->first;
          Matrix var = it->second;
          unsigned agentID = get<0>(nID);
          unsigned localID = get<1>(nID);
          agents[robot2]->updateNeighborPose(0, agentID, localID, var);
        }
      }
    }

    // Randomly select a robot to optimize
    unsigned selectedRobot = (unsigned) distribution(generator);

    // All robots perform an iteration
    for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
      PGOAgent *robotPtr = agents[robot];
      assert(robotPtr->instance_number() == 0);
      assert(robotPtr->iteration_number() == iter);
      if (robot == selectedRobot) {
        robotPtr->iterate(true);
      }else{
        robotPtr->iterate(false);
      }
    }

    // Evaluate cost at this iteration
    for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
      unsigned startIdx = robot * num_poses_per_robot;
      unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
      if (robot == (unsigned) num_robots - 1) endIdx = n;

      Matrix Xrobot;
      agents[robot]->getX(Xrobot);
      Xopt.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) = Xrobot;
    }

    cout << "Iter = " << iter << " | "
         << "cost = " << (Xopt * ConLapT * Xopt.transpose()).trace() << " | "
         << "robot = " << selectedRobot << endl;
  }

  exit(0);
}
