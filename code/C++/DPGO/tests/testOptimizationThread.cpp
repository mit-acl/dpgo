#include "gtest/gtest.h"
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"
#include "DPGO_utils.h"

using namespace DPGO;

TEST(testDPGO, OptimizationThread)
{
    unsigned int d,r;
    d = 3;
    r = 3;
    ROPTALG algorithm = ROPTALG::RTR;
    bool verbose = false;
    PGOAgentParameters options(d,r,algorithm,verbose);

    PGOAgent agent(0, options);

    ASSERT_FALSE(agent.isOptimizationRunning());

    for(unsigned trial = 0; trial < 3; ++trial){
    	agent.startOptimizationLoop(1.0);
	    sleep(1);
	    ASSERT_TRUE(agent.isOptimizationRunning());
	    agent.endOptimizationLoop();
	    ASSERT_FALSE(agent.isOptimizationRunning());
    }
}