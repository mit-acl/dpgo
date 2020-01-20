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
    Matrix Y;
    SparseMatrix B1, B2, B3; 
    construct_B_matrices(dataset, B1, B2, B3);
    Matrix Rinit = chordal_initialization(d, B3);
    Matrix tinit = recover_translations(B1, B2, Rinit);
    Y.resize(r, n*(d+1));
    Y.setZero();
    for (size_t i=0; i<n; i++)
    {
        Y.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Y.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
    }


    // Agent 1 owns pose [0, n1)
    // Agent 2 owns pose [n1, n)
    unsigned int n1 = n/2;

    // initialize two agents
    PGOAgent agent1(0, d, r, false);
    PGOAgent agent2(1, d, r, false);


    for(size_t k = 0; k < dataset.size(); ++k){
        RelativePoseMeasurement mIn = dataset[k];
        if(mIn.i < n1 && mIn.j < n1){
            // private measuerment of agent 1
            RelativeSEMeasurement m(0,0,mIn.i, mIn.j, mIn.R, mIn.t, mIn.kappa, mIn.tau);

            if(mIn.i + 1== mIn.j){
                // odometry
                agent1.addOdometry(m);
            }else{
                // private loop closure
                agent1.addPrivateLoopClosure(m);
            }
        }
        else if (mIn.i >= n1 && mIn.j >= n1){
            // private measurement of agent 2
            RelativeSEMeasurement m(1,1,mIn.i-n1, mIn.j-n1, mIn.R, mIn.t, mIn.kappa, mIn.tau);
            if(mIn.i + 1== mIn.j){
                // odometry
                agent2.addOdometry(m);
            }else{
                // private loop closure
                agent2.addPrivateLoopClosure(m);
            }
        }
        else if (mIn.i < n1 && mIn.j >= n1){
            // shared loop closure from agent 0 to agent 1
            RelativeSEMeasurement m(0,1,mIn.i, mIn.j-n1, mIn.R, mIn.t, mIn.kappa, mIn.tau);
            agent1.addSharedLoopClosure(m);
            agent2.addSharedLoopClosure(m);

        }
        else if (mIn.i >= n1 && mIn.j < n1){
            // shared loop closure from agent 1 to 0
            RelativeSEMeasurement m(1,0,mIn.i-n1, mIn.j, mIn.R, mIn.t, mIn.kappa, mIn.tau);
            agent1.addSharedLoopClosure(m);
            agent2.addSharedLoopClosure(m);
        }

    }


    // Initialize
    agent1.setY(Y.block(0,0,r,n1*(d+1)));
    agent2.setY(Y.block(0,n1*(d+1),r,(n-n1)*(d+1)));
    Matrix Yopt = Y;
    unsigned numIters = 10;
    cout << "Running RBCD for " << numIters << " iterations..." << endl; 
    for (unsigned iter = 0; iter < numIters; ++iter){
        cout 
        << "Iter = " << iter << " | "
        << "cost = " << (Yopt * ConLapT * Yopt.transpose()).trace() << endl;

        // exchange public poses
        PoseDict agent2SharedPoses = agent2.getSharedPoses();
        for(auto it = agent2SharedPoses.begin(); it != agent2SharedPoses.end(); ++it){
            PoseID nID = it->first; 
            Matrix var = it->second;
            unsigned agentID = get<0>(nID);
            unsigned localID = get<1>(nID);
            agent1.updateNeighborPose(0,agentID,localID, var);
        }
        agent1.optimize();

        PoseDict agent1SharedPoses = agent1.getSharedPoses();
        for(auto it = agent1SharedPoses.begin(); it != agent1SharedPoses.end(); ++it){
            PoseID nID = it->first; 
            Matrix var = it->second;
            unsigned agentID = get<0>(nID);
            unsigned localID = get<1>(nID);
            agent2.updateNeighborPose(0,agentID,localID, var);
        }
        agent2.optimize();

        Yopt.block(0,0,r,n1*(d+1)) = agent1.getY();
        Yopt.block(0,n1*(d+1),r,(n-n1)*(d+1)) = agent2.getY();        

    }

    exit(0);
}
