#include <iostream>
#include <cassert>
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
    
    cout << "Testing optimization thread of PGOAgent..." << endl;

    size_t num_poses;
    string TESTFILE = "data/smallGrid3D.g2o";
    vector<SESync::RelativePoseMeasurement> dataset = SESync::read_g2o_file(TESTFILE, num_poses);
    
    if (num_poses <= 0){
    	cout << "Read empty dataset. Does this file exists? " << endl;
    	cout << TESTFILE << endl;
    	exit(1);
    }

    cout << "Read dataset from file " << TESTFILE << endl;
    cout << "Number of poses = " << num_poses << endl;
    
    unsigned int d,r;
    d = (!dataset.empty() ? dataset[0].t.size() : 0);
    r = 5;
    bool verbose = false;
    ROPTALG algorithm = ROPTALG::RTR;
    PGOAgentParameters options(d,r,verbose,algorithm);

    PGOAgent agent(0, options);

    for(size_t k = 0; k < dataset.size(); ++k){
        RelativePoseMeasurement mIn = dataset[k];
        unsigned srcIdx = mIn.i;
        unsigned dstIdx = mIn.j;
        RelativeSEMeasurement m(0, 0, srcIdx, dstIdx, mIn.R, mIn.t, mIn.kappa, mIn.tau);
        // private measurement
        if(srcIdx + 1 == dstIdx){
            // Odometry
            agent.addOdometry(m);
        }
        else{
            // private loop closure
            agent.addPrivateLoopClosure(m);
        }
    }

    cout << "Begin testing... " << endl;

    assert(agent.isOptimizationRunning() == false);

    for(unsigned trial = 0; trial < 5; ++trial){
    	agent.startOptimizationLoop(1.0);
	    sleep(1);
	    assert(agent.isOptimizationRunning() == true);
	    agent.endOptimizationLoop();
	    assert(agent.isOptimizationRunning() == false);
    }

    cout << "Tests passed." << endl;

    exit(0);
}