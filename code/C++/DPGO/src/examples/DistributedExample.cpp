#include <iostream>
#include <fstream>
#include "SESync.h"
#include "SESync_utils.h"
#include "SESync_types.h"
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"

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
    vector<RelativeSEMeasurement> odometry;
    vector<SESync::RelativePoseMeasurement> dataset = SESync::read_g2o_file(argv[1], num_poses);
    cout << "Loaded dataset from file " << argv[1] << endl;
    unsigned d = (!dataset.empty() ? dataset[0].t.size() : 0);
    unsigned r = 5;

    for(size_t i = 0; i < dataset.size(); ++i){
        RelativeSEMeasurement m(0,0,dataset[i].i,dataset[i].j,dataset[i].R,dataset[i].t,dataset[i].kappa, dataset[i].tau);
        odometry.push_back(m);
    }
    unsigned agentID = 0;
    PGOAgent agent(agentID,d,r);
    for(size_t i = 0; i < odometry.size(); ++i){
        agent.addOdometry(odometry[i]);
    }

    Matrix T = agent.getTrajectoryInLocalFrame();


    // Save to file
    string filename = "/home/yulun/git/dpgo/code/results/trajectory.txt";
    ofstream file;
    file.open(filename.c_str(), std::ofstream::out);
    file << T << std::endl;
    file.close();


    exit(0);
}
