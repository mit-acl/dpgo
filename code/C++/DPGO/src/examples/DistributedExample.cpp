#include <iostream>
#include <fstream>
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

int main(int argc, char** argv)
{
    
    if (argc < 2) {
        cout << "Distributed pose-graph optimization. " << endl;
        cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
        exit(1);
    }

    size_t num_poses;
    vector<SESync::RelativePoseMeasurement> dataset = SESync::read_g2o_file(argv[1], num_poses);
    cout << "Loaded dataset from file " << argv[1] << endl;
    
    unsigned int n,d,r;
    SparseMatrix ConLapT = construct_connection_Laplacian_T(dataset);
    d = (!dataset.empty() ? dataset[0].t.size() : 0);
    n = ConLapT.rows()/(d+1);
    r = 5;

    // Chordal initialization
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


    vector<PGOAgent::PoseID> agent1PublicPoses;
    vector<PGOAgent::PoseID> agent2PublicPoses;

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

            agent1PublicPoses.push_back(make_pair(0,mIn.i));
            agent2PublicPoses.push_back(make_pair(1,mIn.j - n1));

        }
        else if (mIn.i >= n1 && mIn.j < n1){
            // shared loop closure from agent 1 to 0
            RelativeSEMeasurement m(1,0,mIn.i-n1, mIn.j, mIn.R, mIn.t, mIn.kappa, mIn.tau);
            agent1.addSharedLoopClosure(m);
            agent2.addSharedLoopClosure(m);

            agent1PublicPoses.push_back(make_pair(0,mIn.j));
            agent2PublicPoses.push_back(make_pair(1,mIn.i-n1));
        }

    }

    cout << "# Agent 1 public poses: " << endl;
    for(size_t i = 0; i < agent1PublicPoses.size() ; ++i){
        cout << get<1>(agent1PublicPoses[i]) << ", ";
    }
    cout << endl;

    cout << "# Agent 2 public poses: " <<  endl;
    for(size_t i = 0; i < agent2PublicPoses.size() ; ++i){
        cout << get<1>(agent2PublicPoses[i]) << ", ";
    }
    cout << endl;


    // Initialize
    agent1.setY(Y.block(0,0,r,n1*(d+1)));
    agent2.setY(Y.block(0,n1*(d+1),r,(n-n1)*(d+1)));

    cout << "COST = " << (Y * ConLapT * Y.transpose()).trace() << endl;

    unsigned numIters = 500;
    for (unsigned iter = 0; iter < numIters; ++iter){
        // exchange public poses
        Matrix Y2 = agent2.getY();
        for(size_t i = 0; i < agent2PublicPoses.size() ; ++i){
            PGOAgent::PoseID pose_id = agent2PublicPoses[i];
            assert(get<0>(pose_id) == 1);
            unsigned idx = get<1>(pose_id);
            Matrix Yi = Y2.block(0, idx*(d+1), r, d+1);
            agent1.updateSharedPose(0,1,idx,Yi);
        }
        agent1.optimize();


        Matrix Y1 = agent1.getY();
        for(size_t i = 0; i < agent1PublicPoses.size(); ++i){
            PGOAgent::PoseID pose_id = agent1PublicPoses[i];
            assert(get<0>(pose_id) == 0);
            unsigned idx = get<1>(pose_id);
            Matrix Yi = Y1.block(0, idx*(d+1), r, d+1);
            agent2.updateSharedPose(0,0,idx,Yi);
        }
        agent2.optimize();

        Matrix Yopt = Matrix(r, n*(d+1));
        Yopt.block(0,0,r,n1*(d+1)) = agent1.getY();
        Yopt.block(0,n1*(d+1),r,(n-n1)*(d+1)) = agent2.getY();

        cout << "COST = " << (Yopt * ConLapT * Yopt.transpose()).trace() << endl;

    }

    exit(0);
}
