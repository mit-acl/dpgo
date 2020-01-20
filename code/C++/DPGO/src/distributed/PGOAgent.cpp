#include "distributed/PGOAgent.h"
#include <iostream>
#include <cassert>
#include <chrono>
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
	verbose(true)
	{
		// automatically initialize the first pose on the Cartan group
		LiftedSEVariable x(r,d,1);
		x.var()->RandInManifold();
		Y = x.getData();
	}

	PGOAgent::PGOAgent(unsigned ID, unsigned dIn, unsigned rIn, bool v):
	mID(ID), 
	mCluster(ID), 
	d(dIn), 
	r(rIn), 
	n(1),
	verbose(v)
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
		lock_guard<mutex> tLock(mPosesMutex);
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
		if(verbose) cout << "Initialized pose " << n << endl;

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
		if(verbose) cout << "Add private loop closure " << privateLoopClosures.size() << endl;
	}

	void PGOAgent::addSharedLoopClosure(const RelativeSEMeasurement& factor){
		if(factor.r1 == mID){
			assert(factor.p1 < n);
			assert(factor.r2 != mID);
			mSharedPoses.insert(make_pair(mID, factor.p1));
			neighborSharedPoses.insert(make_pair(factor.r2,factor.p2));
		}
		else{
			assert(factor.r2 == mID);
			assert(factor.p2 < n);
			mSharedPoses.insert(make_pair(mID, factor.p2));
			neighborSharedPoses.insert(make_pair(factor.r1, factor.p1));
		}

		lock_guard<mutex> lock(mMeasurementsMutex);
		sharedLoopClosures.push_back(factor);
		if(verbose) cout << "Add shared loop closure " << sharedLoopClosures.size() << endl;
	}

	void PGOAgent::updateNeighborPose(unsigned neighborCluster, unsigned neighborID, unsigned neighborPose, const Matrix& var){
		assert(neighborID != mID);

		/** 
        TODO: if necessary, realign the local frame of this robot to match the neighbor's
        and update the cluster that this robot belongs to
    	*/

		PoseID nID = std::make_pair(neighborID, neighborPose);

		// Do not store this pose if not needed
		if(neighborSharedPoses.find(nID) == neighborSharedPoses.end()) return;

		lock_guard<mutex> lock(mNeighborPosesMutex);

		neighborPoseDict[nID] = var;
	}


	Matrix PGOAgent::getTrajectoryInLocalFrame(){
		lock_guard<mutex> lock(mPosesMutex);

		Matrix T = Y.block(0,0,r,d).transpose() * Y;
		Matrix t0 = T.block(0,d,d,1);

		for(unsigned i = 0; i < n; ++i){
			T.block(0,i*(d+1),d,d) = projectToRotationGroup(T.block(0,i*(d+1),d,d));
			T.block(0,i*(d+1)+d,d,1) = T.block(0,i*(d+1)+d,d,1) - t0;
		}

		return T;
	}


	PoseDict PGOAgent::getSharedPoses(){
		PoseDict map;
		lock_guard<mutex> lock(mPosesMutex);
		for(auto it = mSharedPoses.begin(); it!= mSharedPoses.end(); ++it){
			unsigned idx = get<1>(*it);
			map[*it] = Y.block(0, idx*(d+1), r, d+1);
		}
		return map;
	}


	void PGOAgent::optimize(){
		if(verbose) cout << "Agent " << mID << " optimize..." << endl;

		// number of poses updated at this time
		unsigned k = n; 

		vector<RelativeSEMeasurement> myMeasurements;
		unique_lock<mutex> mLock(mMeasurementsMutex);
		for(size_t i = 0; i < odometry.size(); ++i){
			RelativeSEMeasurement m = odometry[i];
			if(m.p1 < k && m.p2 < k) myMeasurements.push_back(m);
		}
		for(size_t i = 0; i < privateLoopClosures.size(); ++i){
			RelativeSEMeasurement m = privateLoopClosures[i];
			if(m.p1 < k && m.p2 < k) myMeasurements.push_back(m);
		}
		mLock.unlock();
		if (myMeasurements.empty()) return;

		vector<RelativeSEMeasurement> sharedMeasurements;
		for(size_t i = 0; i < sharedLoopClosures.size(); ++i){
			RelativeSEMeasurement m = sharedLoopClosures[i];
			if(m.r1 == mID && m.p1 < k) sharedMeasurements.push_back(m);
			else if(m.r2 == mID && m.p2 < k) sharedMeasurements.push_back(m);
		}

		SparseMatrix Q((d+1)*k, (d+1)*k);
		SparseMatrix G(r,(d+1)*k);
		constructCostMatrices(myMeasurements, sharedMeasurements, &Q, &G);

		// Read current estimates of the first k poses
		unique_lock<mutex> tLock(mPosesMutex);
		Matrix Ycurr = Y.block(0,0,r,(d+1)*k);
		tLock.unlock();
		assert(Ycurr.cols() == Q.cols());

		// Construct optimization problem
		QuadraticProblem problem(k, d, r, Q, G);
		if (verbose) cout << "Constructed quadratic problem instance." << endl;
		
		// Initialize optimizer object
		QuadraticOptimizer optimizer(&problem);
		optimizer.setVerbose(verbose);
		if (verbose) cout << "Constructed optimizer. Optimizing..." << endl;
		
		// Optimize
		auto startTime = std::chrono::high_resolution_clock::now();
		Matrix Ynext = optimizer.optimize(Ycurr);
		auto counter = std::chrono::high_resolution_clock::now() - startTime;
		double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();
		if (verbose) cout << "Optimization time: " << elapsedMs / 1000 << " seconds." << endl;

		// TODO: handle dynamic pose graph
		tLock.lock();
		Y = Ynext;
		tLock.unlock();
	}


	void PGOAgent::constructCostMatrices(const vector<RelativeSEMeasurement>& privateMeasurements,
            const vector<RelativeSEMeasurement>& sharedMeasurements,
            SparseMatrix* Q, 
            SparseMatrix* G)
	{
		*Q = constructConnectionLaplacianSE(privateMeasurements);

		unique_lock<mutex> lock(mNeighborPosesMutex, std::defer_lock);

		// go through shared measurements
		for(size_t i = 0; i < sharedMeasurements.size(); ++i){
			RelativeSEMeasurement m = sharedMeasurements[i];

			// Construct relative SE matrix in homogeneous form
			Matrix T = Matrix::Zero(d+1,d+1);
			T.block(0,0,d,d) = m.R;
			T.block(0,d,d,1) = m.t;
			T(d,d) = 1;


			// Construct aggregate weight matrix
			Matrix Omega = Matrix::Zero(d+1,d+1);
			for(unsigned row = 0; row < d; ++row){
				Omega(row,row) = m.kappa;
			}
			Omega(d,d) = m.tau;


			if(m.r1 == mID){
				// First pose belongs to this robot
				// Hence, this is an outgoing edge in the pose graph
				assert(m.r2 != mID);

				// Read neighbor's pose
				const PoseID nID = make_pair(m.r2, m.p2);
				auto KVpair = neighborPoseDict.find(nID);
				if(KVpair == neighborPoseDict.end()){
					cout << "WARNING: shared pose does not exist!" << endl;
					continue;
				}
				lock.lock();
				Matrix Yj = KVpair->second;
				lock.unlock();

				// Modify quadratic cost
				size_t idx = m.p1;
				Matrix W = T * Omega * T.transpose();
				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < d+1; ++row){
						Q->coeffRef(idx*(d+1)+row, idx*(d+1)+col) += W(row,col);
					}
				}

				// Modify linear cost 
				Matrix L = - Yj * Omega * T.transpose();
				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < r; ++row){
						G->coeffRef(row, idx*(d+1)+col) += L(row,col);
					}
				}


			}
			else{
				// Second pose belongs to this robot
				// Hence, this is an incoming edge in the pose graph
				assert(m.r2 == mID);

				// Read neighbor's pose
				const PoseID nID = make_pair(m.r1, m.p1);
				auto KVpair = neighborPoseDict.find(nID);
				if(KVpair == neighborPoseDict.end()){
					cout << "WARNING: shared pose does not exist!" << endl;
					continue;
				}
				lock.lock();
				Matrix Yi = KVpair->second;
				lock.unlock();

				// Modify quadratic cost
				size_t idx = m.p2;
				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < d+1; ++row){
						Q->coeffRef(idx*(d+1)+row, idx*(d+1)+col) += Omega(row,col);
					}
				}

				// Modify linear cost
				Matrix L = - Yi * T * Omega;
				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < r; ++row){
						G->coeffRef(row, idx*(d+1)+col) += L(row,col);
					}
				}

			}

		}

	}


}