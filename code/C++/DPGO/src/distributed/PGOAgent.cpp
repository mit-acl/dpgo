#include "distributed/PGOAgent.h"
#include <iostream>
#include <cassert>
#include "DPGO_utils.h"

using namespace std;

namespace DPGO{

	PGOAgent::PGOAgent(unsigned ID, unsigned dIn, unsigned rIn): mID(ID), mCluster(ID), d(dIn), r(rIn), n(1){
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

		odometry.push_back(factor);

		// extend trajectory by a single pose
		lock_guard<mutex> lock(mTrajectoryMutex);
		Y.conservativeResize(Eigen::NoChange, (d+1)*(n+1));

		Matrix currR = Y.block(0, (n-1)*(d+1), r, d);
		Matrix currt = Y.block(0, (n-1)*(d+1)+d,r,1);

		// initialize next pose by propagating odometry
		Matrix nextR = currR * factor.R;
		Matrix nextt = currt + currR * factor.t;
		Y.block(0, n*(d+1), r, d) = nextR;
		Y.block(0, n*(d+1)+d,r,1) = nextt;

		n++;
		assert((d+1)*n == Y.cols());
		cout << "Length of trajectory: " << n << endl;
	}

	void PGOAgent::updateSharedPose(unsigned neighborCluster, unsigned neighborID, unsigned neighborPose, const Matrix& var){
		assert(neighborID != mID);

		/** 
        TODO: if necessary, realign the local frame of this robot to match the neighbor's
        and update the cluster that this robot belongs to
    	*/

		PoseID nID = std::make_pair(neighborID, neighborPose);

		lock_guard<mutex> lock(mSharedPosesMutex);

		sharedPoseDict[nID] = var;
	}


	Matrix PGOAgent::getTrajectoryInLocalFrame(){
		Matrix T = Y.block(0,0,r,d).transpose() * Y;
		Matrix t0 = T.block(0,d,d,1);

		for(unsigned i = 0; i < n; ++i){
			T.block(0,i*(d+1),d,d) = projectToRotationGroup(T.block(0,i*(d+1),d,d));
			T.block(0,i*(d+1)+d,d,1) = T.block(0,i*(d+1)+d,d,1) - t0;
		}

		return T;
	}


}