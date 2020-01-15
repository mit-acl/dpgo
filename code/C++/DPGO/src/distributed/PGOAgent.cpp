#include "distributed/PGOAgent.h"
#include <iostream>
#include <cassert>

using namespace std;

namespace DPGO{

	PGOAgent::PGOAgent(unsigned ID, unsigned dIn, unsigned rIn): mID(ID), d(dIn), r(rIn), n(1){
		// automatically initialize the first pose on the Cartan group
		LiftedSEVariable x(r,d,1);
		x.var()->RandInManifold();
		Y = x.getData();
	}

	PGOAgent::~PGOAgent(){}

	void PGOAgent::addOdometry(const RelativeSEMeasurement& factor){
		// check that this is a odometric measurement 
		assert(factor.r1 == mID);
		assert(factor.r1 == factor.r2);
		assert(factor.p1 == n-1);
		assert(factor.p2 == n);

		// extend trajectory by a single pose
		lock_guard<mutex> lock(mTrajectoryMutex);
		// TODO!
		n++;
		assert((d+1)*n == Y.cols());
	}

	void PGOAgent::updateNeighborPose(unsigned agent, unsigned pose, const Matrix& var){
		assert(agent != mID);

		PoseID pID = std::make_pair(agent, pose);

		cachedNeighborPoses[pID] = var;
	}


}