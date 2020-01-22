#include <iostream>
#include <fstream>
#include <cstdlib>
#include "SESync.h"
#include "SESync_utils.h"
#include "SESync_types.h"
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"
#include "DPGO_utils.h"

using namespace std;
using namespace DPGO;
using namespace SESync;


/**
This demo simulates a serial version of the distributed PGO algorithm described in:

Y. Tian, K. Khosoussi, and JP How
"Block-Coordinate Descent on the Riemannian Staircase for Certifiably Correct Distributed Rotation and Pose Synchronization"
*/

int main(int argc, char** argv)
{
    
    if (argc < 3) {
        cout << "Distributed pose-graph optimization. " << endl;
        cout << "Usage: " << argv[0] << " [# robots] [input .g2o file]" << endl;
        exit(1);
    }

    cout << "Distributed pose-graph optimization demo. " << endl;

    int num_robots = atoi(argv[1]);
    if (num_robots <= 0){
        cout << "Number of robots must be positive!" << endl;
        exit(1);
    }
    cout << "Simulating " << num_robots << " robots." << endl;


    size_t num_poses;
    vector<SESync::RelativePoseMeasurement> dataset = SESync::read_g2o_file(argv[2], num_poses);
    cout << "Loaded dataset from file " << argv[2] << "." << endl;
    
    unsigned int n,d,r;
    SparseMatrix ConLapT = construct_connection_Laplacian_T(dataset);
    d = (!dataset.empty() ? dataset[0].t.size() : 0);
    n = num_poses;
    r = 5;

    // We use SE-Sync's implementation of chordal initialization
    Matrix Yinit;
    SparseMatrix B1, B2, B3; 
    construct_B_matrices(dataset, B1, B2, B3);
    Matrix Rinit = chordal_initialization(d, B3);
    Matrix tinit = recover_translations(B1, B2, Rinit);
    Yinit.resize(r, n*(d+1));
    Yinit.setZero();
    for (size_t i=0; i<n; i++)
    {
        Yinit.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Yinit.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
    }

    unsigned int num_poses_per_robot = n/num_robots;
    if(num_poses_per_robot <= 0){
        cout << "More robots than total number of poses! Decrease the number of robots" << endl;
        exit(1);
    }

    // create mapping from global pose index to local pose index
    map<unsigned, PoseID> PoseMap;
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        // cout << "Poses for robot " << robot << endl;
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned) num_robots - 1) endIdx = n;
        for(unsigned idx = startIdx; idx < endIdx; ++idx){
            unsigned localIdx = idx - startIdx; // this is the local ID of this pose
            PoseID pose = make_pair(robot, localIdx);
            PoseMap[idx] = pose;
            // cout << idx << ", ";
        }
        cout << endl;
    }

    
    vector<PGOAgent*> agents;
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        PGOAgent* ag = new PGOAgent(robot, d, r, false);
        agents.push_back(ag);
    }


    for(size_t k = 0; k < dataset.size(); ++k){
        RelativePoseMeasurement mIn = dataset[k];
        PoseID src = PoseMap[mIn.i];
        PoseID dst = PoseMap[mIn.j];

        unsigned srcRobot = get<0>(src);
        unsigned srcIdx = get<1>(src);
        unsigned dstRobot = get<0>(dst);
        unsigned dstIdx = get<1>(dst);

        RelativeSEMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx, mIn.R, mIn.t, mIn.kappa, mIn.tau);

        if (srcRobot == dstRobot){
            // private measurement
            if(srcIdx + 1 == dstIdx){
                // Odometry
                agents[srcRobot]->addOdometry(m);
            }
            else{
                // private loop closure
                agents[srcRobot]->addPrivateLoopClosure(m);
            }
        }else{
            // shared measurement
            agents[srcRobot]->addSharedLoopClosure(m);
            agents[dstRobot]->addSharedLoopClosure(m);
        }

    }


    cout << "Initializing..." << endl;
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned) num_robots - 1) endIdx = n;

        agents[robot]->setY(Yinit.block(0, startIdx*(d+1), r, (endIdx-startIdx)*(d+1)));
        
    }


    Matrix Yopt = Yinit;
    unsigned numIters = 1000;
    cout << "Running RBCD for " << numIters << " iterations..." << endl; 


    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,num_robots-1);

    for (unsigned iter = 0; iter < numIters; ++iter){
        // exchange public poses
        for(unsigned robot1 = 0; robot1 < (unsigned) num_robots; ++robot1){
            PoseDict sharedPoses = agents[robot1]->getSharedPoses();
            for(unsigned robot2 = 0; robot2 < (unsigned) num_robots; ++robot2){
                if(robot1 == robot2) continue;
                
                for(auto it = sharedPoses.begin(); it != sharedPoses.end(); ++it){
                    PoseID nID = it->first; 
                    Matrix var = it->second;
                    unsigned agentID = get<0>(nID);
                    unsigned localID = get<1>(nID);
                    agents[robot2]->updateNeighborPose(0,agentID,localID, var);
                }

            }
        }


        // randomly select a robot to optimize
        unsigned robot = (unsigned) distribution(generator);
        agents[robot]->optimize();

        
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned) num_robots - 1) endIdx = n;
        Yopt.block(0, startIdx*(d+1), r, (endIdx-startIdx)*(d+1)) = agents[robot]->getY();

        // Evaluate
        cout 
        << "Iter = " << iter << " | "
        << "cost = " << (Yopt * ConLapT * Yopt.transpose()).trace() << " | "
        << "robot = " << robot << endl;
    }

    exit(0);
}
