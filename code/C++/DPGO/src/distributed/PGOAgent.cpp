#include "distributed/PGOAgent.h"
#include <iostream>
#include <cassert>
#include "DPGO_utils.h"
#include "QuadraticProblem.h"
#include "QuadraticOptimizer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>

using namespace std;

namespace DPGO{

	PGOAgent::PGOAgent(unsigned ID, unsigned dIn, unsigned rIn): 
	mID(ID), 
	mCluster(ID), 
	d(dIn), 
	r(rIn), 
	n(1),
	maxStepsize(0.001),
	stepsizeDecay(0.8),
	maxLineSearchAttempts(100)
	{
		// automatically initialize the first pose on the Cartan group
		LiftedSEVariable x(r,d,1);
		x.var()->RandInManifold();
		Y = x.getData();
	}

	PGOAgent::~PGOAgent(){}

	void PGOAgent::addOdometry(const RelativeSEMeasurement& factor){
		// check that this is a odometric measurement 
		assert(factor.r1 == mID);
		assert(factor.r2 == mID);
		assert(factor.p1 == n-1);
		assert(factor.p2 == n);

		// extend trajectory by a single pose
		lock_guard<mutex> tLock(mTrajectoryMutex);
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
		cout << "Initialized pose " << n << endl;

		lock_guard<mutex> mLock(mMeasurementsMutex);
		odometry.push_back(factor);
	}

	void PGOAgent::addPrivateLoopClosure(const RelativeSEMeasurement& factor){
		assert(factor.r1 == mID);
		assert(factor.r2 == mID);
		assert(factor.p1 < n);
		assert(factor.p2 < n);

		lock_guard<mutex> lock(mMeasurementsMutex);
		privateLoopClosures.push_back(factor);
		cout << "Add private loop closure " << privateLoopClosures.size() << endl;
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

	void PGOAgent::optimize(){
		unique_lock<mutex> mLock(mMeasurementsMutex);
		vector<RelativeSEMeasurement> measurements = odometry;
		measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());
		mLock.unlock();
		if (measurements.empty()) return;

		// Compute connection laplacian matrix (all private factors)
		SparseMatrix Q = constructConnectionLaplacianSE(measurements);

		// Number of poses included in this optimization
		unsigned k = Q.rows() / (d+1);
		unique_lock<mutex> tLock(mTrajectoryMutex);
		Matrix Ycurr = Y.block(0,0,r,(d+1)*k);
		tLock.unlock();
		assert(Ycurr.cols() == Q.cols());

		// TODO: compute linear factors (all shared factors)
		SparseMatrix G(r,(d+1)*k);
    	G.setZero();


		QuadraticProblem problem(k, d, r, Q, G);
		cout << "Constructed quadratic problem instance." << endl;
		QuadraticOptimizer optimizer(&problem);
		cout << "Constructed optimizer." << endl;
		Matrix Ynext = optimizer.optimize(Ycurr);


		// TODO: handle dynamic pose graph
		tLock.lock();
		Y = Ynext;
		tLock.unlock();
	}


}