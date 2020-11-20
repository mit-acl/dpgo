
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
    cout << "Multi-robot pose graph optimization example. " << endl;
    cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]" << endl;
    exit(1);
  }

  cout << "Multi-robot pose graph optimization example. " << endl;

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
  Options
  ###########################################
  */
  unsigned int n, d, r;
  d = (!dataset.empty() ? dataset[0].t.size() : 0);
  n = num_poses;
  r = 5;
  bool acceleration = false;
  unsigned restartInterval = 30;
  unsigned numIters = 1000;
  bool centralized_chordal_initialization = false;

  // Construct the centralized problem (used for evaluation)
  SparseMatrix QCentral = constructConnectionLaplacianSE(dataset);
  SparseMatrix GCentral(r, (d + 1) * n);
  QuadraticProblem problemCentral(n, d, r, QCentral, GCentral);


  /**
  ###########################################
  Partition dataset into robots
  ###########################################
  */
  unsigned int num_poses_per_robot = num_poses / num_robots;
  if (num_poses_per_robot <= 0) {
    cout << "More robots than total number of poses! Decrease the number of robots" << endl;
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
    PGOAgentParameters options(d, r, num_robots, acceleration, restartInterval);
    std::cout << options << std::endl;

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
  ############################################
  Optional: centralized chordal initialization
  ############################################
  */
  if (centralized_chordal_initialization) {
    Matrix TChordal = chordalInitialization(d, n, dataset);
    Matrix XChordal = fixedStiefelVariable(d, r) * TChordal;
    std::cout << "Chordal initialization cost = " << problemCentral.f(XChordal) << std::endl;
    for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
      unsigned startIdx = robot * num_poses_per_robot;
      unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
      if (robot == (unsigned) num_robots - 1) endIdx = n;
      agents[robot]->setX(XChordal.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)));
    }
  }

  /**
  ###########################################
  Optimization loop
  ###########################################
  */
  Matrix Xopt(r, n * (d + 1));
  unsigned selectedRobot = 0;
  cout << "Running " << numIters << " iterations..." << endl;
  for (unsigned iter = 0; iter < numIters; ++iter) {
    PGOAgent *selectedRobotPtr = agents[selectedRobot];

    // Evaluate cost at this iteration
    bool all_initialized = true;
    for (unsigned robot = 0; robot < (unsigned) num_robots; ++robot) {
      unsigned startIdx = robot * num_poses_per_robot;
      unsigned endIdx = (robot + 1) * num_poses_per_robot;  // non-inclusive
      if (robot == (unsigned) num_robots - 1) endIdx = n;

      Matrix Xrobot;
      if (agents[robot]->getX(Xrobot)) {
        Xopt.block(0, startIdx * (d + 1), r, (endIdx - startIdx) * (d + 1)) = Xrobot;
      }
      else{
        all_initialized = false;
        break;
      }
    }
    if (all_initialized)
      std::cout << std::setprecision(5)
                << "Iter = " << iter << " | "
                << "robot = " << selectedRobotPtr->getID() << " | "
                << "cost = " << 2 * problemCentral.f(Xopt) << " | "
                << "gradnorm = " << problemCentral.gradNorm(Xopt) << std::endl;


    // Non-selected robots perform an iteration
    for (auto *robotPtr : agents) {
      assert(robotPtr->instance_number() == 0);
      assert(robotPtr->iteration_number() == iter);
      if (robotPtr->getID() != selectedRobot) {
        robotPtr->iterate(false);
      }
    }

    // Selected robot requests public poses from others
    for (auto *robotPtr : agents) {
      if (robotPtr->getID() == selectedRobot) continue;
      PoseDict sharedPoses;
      if (!robotPtr->getSharedPoseDict(sharedPoses)) {
        continue;
      }
      for (auto it = sharedPoses.begin(); it != sharedPoses.end(); ++it) {
        PoseID nID = it->first;
        Matrix var = it->second;
        unsigned agentID = get<0>(nID);
        unsigned localID = get<1>(nID);
        selectedRobotPtr->updateNeighborPose(0, agentID, localID, var);
      }
    }

    // When using acceleration, selected robot also requests auxiliary poses
    if (acceleration) {
      for (auto *robotPtr : agents) {
        if (robotPtr->getID() == selectedRobot) continue;
        PoseDict auxSharedPoses;
        if (!robotPtr->getAuxSharedPoseDict(auxSharedPoses)) {
          continue;
        }
        for (auto it = auxSharedPoses.begin(); it != auxSharedPoses.end(); ++it) {
          PoseID nID = it->first;
          Matrix var = it->second;
          unsigned agentID = get<0>(nID);
          unsigned localID = get<1>(nID);
          selectedRobotPtr->updateAuxNeighborPose(0, agentID, localID, var);
        }
      }
    }

    // Selected robot update
    selectedRobotPtr->iterate(true);

    // Select next robot
    std::vector<unsigned> neighbors = selectedRobotPtr->getNeighbors();
    std::uniform_int_distribution<int> distribution(0, neighbors.size() - 1);
    std::random_device rd;
    std::mt19937 gen(rd());
    selectedRobot = neighbors[distribution(gen)];

  }

  exit(0);
}
