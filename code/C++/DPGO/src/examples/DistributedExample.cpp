#include <iostream>
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
    PGOAgent agent(0,3,5);

    Matrix M(3,5);
    M.setZero();
    agent.updateNeighborPose(1,1,M);
    agent.updateNeighborPose(1,2,M);
    agent.updateNeighborPose(1,3,M);
    agent.updateNeighborPose(1,3,M);


    exit(0);
}
