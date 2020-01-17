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
    
    unsigned d = (!dataset.empty() ? dataset[0].t.size() : 0);
    unsigned r = 5;
    unsigned agentID = 0;
    PGOAgent agent(agentID,d,r);


    for(size_t k = 0; k < dataset.size(); ++k){
        RelativeSEMeasurement m(0,0,dataset[k].i,dataset[k].j,dataset[k].R,dataset[k].t,dataset[k].kappa, dataset[k].tau);
        if(dataset[k].j == dataset[k].i + 1){
            agent.addOdometry(m);
        }else{
            agent.addPrivateLoopClosure(m);
        }
    }
    
    for (unsigned i = 0; i < 25; ++i){
        agent.optimize();
    }

    // Save to file
    string filename = "/home/yulun/git/dpgo/code/results/trajectory.txt";
    ofstream file;
    file.open(filename.c_str(), std::ofstream::out);
    file << agent.getTrajectoryInLocalFrame() << std::endl;
    file.close();


    exit(0);
}
