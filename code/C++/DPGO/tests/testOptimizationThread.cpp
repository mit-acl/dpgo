/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <iostream>
#include <cassert>
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"
#include "DPGO_utils.h"

using namespace std;
using namespace DPGO;

int main(int argc, char** argv)
{
    
    cout << "Testing optimization thread of PGOAgent..." << endl;
    
    unsigned int d,r;
    d = 3;
    r = 5;
    ROPTALG algorithm = ROPTALG::RTR;
    bool verbose = false;
    PGOAgentParameters options(d,r,algorithm,verbose);

    PGOAgent agent(0, options);


    assert(agent.isOptimizationRunning() == false);

    for(unsigned trial = 0; trial < 5; ++trial){
    	agent.startOptimizationLoop(1.0);
	    sleep(5);
	    assert(agent.isOptimizationRunning() == true);
	    agent.endOptimizationLoop();
	    assert(agent.isOptimizationRunning() == false);
    }

    cout << "Passed." << endl;

    exit(0);
}