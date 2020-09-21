/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/PGOAgent.h>

#include <algorithm>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>
#include <random>

#include <DPGO/DPGO_utils.h>
#include <DPGO/QuadraticProblem.h>
#include <DPGO/QuadraticOptimizer.h>

#include "StieVariable.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>

using namespace std;

namespace DPGO{

	PGOAgent::PGOAgent(unsigned ID, const PGOAgentParameters& params): 
	mID(ID), 
	mCluster(ID), 
	d(params.d), 
	r(params.r), 
	n(1),
	verbose(params.verbose),
	rate(1),
	algorithm(params.algorithm),
	stepsize(1e-3)
	{

		X = Matrix::Zero(r, d+1);
		X.block(0,0,d,d) = Matrix::Identity(d,d);

		// initialize globalAnchor
		globalAnchor = X;
	}


	PGOAgent::~PGOAgent(){
		// Make sure that optimization thread is not running, before exiting
		endOptimizationLoop();
	}


	void PGOAgent::initialize(const std::vector<RelativeSEMeasurement>& inputOdometry, 
                  			  const std::vector<RelativeSEMeasurement>& inputPrivateLoopClosures,
                 			  const std::vector<RelativeSEMeasurement>& inputSharedLoopClosures)
	{
		assert(!isOptimizationRunning());
		assert(n == 1);

		for (size_t i = 0; i < inputOdometry.size(); ++i){
			addOdometry(inputOdometry[i]);
		}
		for (size_t i = 0; i < inputPrivateLoopClosures.size(); ++i){
			addPrivateLoopClosure(inputPrivateLoopClosures[i]);
		}
		for (size_t i = 0; i < inputSharedLoopClosures.size(); ++i){
			addSharedLoopClosure(inputSharedLoopClosures[i]);
		}

		Matrix T = computeInitialEstimate();
		// Lift to correct relaxation rank
		X = fixedStiefelVariable(d,r) * T;
	}


	Matrix PGOAgent::computeInitialEstimate(){
		assert(!isOptimizationRunning());

		std::vector<RelativeSEMeasurement> measurements = odometry;
		measurements.insert(measurements.end(), privateLoopClosures.begin(), privateLoopClosures.end());

		SparseMatrix B1, B2, B3; 
	    constructBMatrices(measurements, B1, B2, B3);
	    Matrix Rinit = chordalInitialization(d, B3);
	    Matrix tinit = recoverTranslations(B1, B2, Rinit);

	    assert(Rinit.rows() == d);
	    assert(Rinit.cols() == d*n);
	    assert(tinit.rows() == d);
	    assert(tinit.cols() == n);

	    Matrix T(d, n*(d+1));
	    for (size_t i=0; i<n; i++)
	    {
	        T.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
	        T.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
	    }

	    return T;
	}


	void PGOAgent::addOdometry(const RelativeSEMeasurement& factor){
		// check that this is a odometric measurement 
		assert(factor.r1 == mID);
		assert(factor.r2 == mID);
		assert(factor.p1 == n-1);
		assert(factor.p2 == n);
		assert(factor.R.rows() == d && factor.R.cols() == d);
		assert(factor.t.rows() == d && factor.t.cols() == 1);

		lock_guard<mutex> tLock(mPosesMutex);
		lock_guard<mutex> nLock(mNeighborPosesMutex);
		
		Matrix X_ = X;
		assert(X_.cols() == (d+1)*n);
		assert(X_.rows() == r);
		X = Matrix::Zero(r,(d+1)*(n+1));
		X.block(0,0,r,(d+1)*n) = X_;

		Matrix currR = X.block(0, (n-1)*(d+1), r, d);
		Matrix currt = X.block(0, (n-1)*(d+1)+d,r,1);

		// initialize next pose by propagating odometry
		Matrix nextR = currR * factor.R;
		Matrix nextt = currt + currR * factor.t;
		X.block(0, n*(d+1), r, d) = nextR;
		X.block(0, n*(d+1)+d,r,1) = nextt;

		n++;
		assert((d+1)*n == X.cols());

		lock_guard<mutex> mLock(mMeasurementsMutex);
		odometry.push_back(factor);
	}


	void PGOAgent::addPrivateLoopClosure(const RelativeSEMeasurement& factor){
		assert(factor.r1 == mID);
		assert(factor.r2 == mID);
		assert(factor.p1 < n);
		assert(factor.p2 < n);
		assert(factor.R.rows() == d && factor.R.cols() == d);
		assert(factor.t.rows() == d && factor.t.cols() == 1);

		lock_guard<mutex> lock(mMeasurementsMutex);
		privateLoopClosures.push_back(factor);
	}


	void PGOAgent::addSharedLoopClosure(const RelativeSEMeasurement& factor){
		assert(factor.R.rows() == d && factor.R.cols() == d);
		assert(factor.t.rows() == d && factor.t.cols() == 1);

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
	}


	void PGOAgent::updateNeighborPose(unsigned neighborCluster, unsigned neighborID, unsigned neighborPose, const Matrix& var){
		assert(neighborID != mID);
		assert(var.rows() == r);
		assert(var.cols() == d+1);

		PoseID nID = std::make_pair(neighborID, neighborPose);

		// Do not store this pose if not needed
		if(neighborSharedPoses.find(nID) == neighborSharedPoses.end()) return;

		/** 
        If necessary, realign the local frame of this robot to match the neighbor's
        and update the cluster that this robot belongs to
    	*/
		if(neighborCluster < mCluster){
			// TODO: initialize again HERE

			cout << "Agent " << mID << " informed by agent " << neighborID << " to join cluster " << neighborCluster << "!" << endl;

			// Halt optimization
			bool optimizationHalted = false;
			if (isOptimizationRunning()){
				if(verbose) cout << "Agent " << mID << " halt optimization thread..." << endl;
				optimizationHalted = true;
				endOptimizationLoop();
			}

			// Halt insertion of new poses
			lock_guard<mutex> tLock(mPosesMutex);
			assert(X.cols() == n*(d+1));

			// Halt insertion of new measurements
			lock_guard<mutex> mLock(mMeasurementsMutex);

			// Clear cache
			lock_guard<mutex> nLock(mNeighborPosesMutex);
			neighborPoseDict.clear();

			mCluster = neighborCluster;

			// Find the corresponding inter-robot loop closure
			RelativeSEMeasurement m;
			assert(findSharedLoopClosure(neighborID, neighborPose, m));

			// Notations:
			// world1: world frame before alignment
			// world2: world frame after alignment 
			// frame1 : frame associated to my public pose
			// frame2 : frame associated to neighbor's public pose
			Matrix dT = Matrix::Identity(d+1, d+1);
			dT.block(0,0,d,d) = m.R;
			dT.block(0,d,d,1) = m.t;

			// Form relative transformation matrix in homogeneous form
			Matrix T_world2_frame2 = Matrix::Identity(d+1,d+1);
			T_world2_frame2.block(0,0,d,d+1) = var;
			Matrix T_frame1_frame2 = Matrix::Identity(d+1,d+1);
			Matrix T_world1_frame1 = Matrix::Identity(d+1,d+1);
			if(m.r1 == neighborID){
				// Incoming edge
				T_frame1_frame2 = dT.inverse();
				T_world1_frame1.block(0,0,d,d+1) = X.block(0, m.p2*(d+1), d, d+1);
			}
			else{
				// Outgoing edge
				T_frame1_frame2 = dT;
				T_world1_frame1.block(0,0,d,d+1) = X.block(0, m.p1*(d+1), d, d+1);
			}
			Matrix T_world2_frame1 = T_world2_frame2 * T_frame1_frame2.inverse();
			Matrix T_world2_world1 = T_world2_frame1 * T_world1_frame1.inverse();

			Matrix T_world1_frame = Matrix::Identity(d+1, d+1);
			Matrix T_world2_frame = Matrix::Identity(d+1, d+1);
			for(size_t i = 0; i < n; ++i){
				T_world1_frame.block(0,0,d,d+1) = X.block(0, i*(d+1), d, d+1);
				T_world2_frame = T_world2_world1 * T_world1_frame;
				X.block(0, i*(d+1), d, d+1) = T_world2_frame.block(0,0,d,d+1);
			}
			
			if(optimizationHalted) startOptimizationLoop(rate);
		}

		// Do not store this pose if it comes from a different cluster
		if(neighborCluster != mCluster) return;

		lock_guard<mutex> lock(mNeighborPosesMutex);
		neighborPoseDict[nID] = var;
	}


	Matrix PGOAgent::getTrajectoryInLocalFrame(){
		lock_guard<mutex> lock(mPosesMutex);

		Matrix T = X.block(0,0,r,d).transpose() * X;
		Matrix t0 = T.block(0,d,d,1);

		for(unsigned i = 0; i < n; ++i){
			T.block(0,i*(d+1),d,d) = projectToRotationGroup(T.block(0,i*(d+1),d,d));
			T.block(0,i*(d+1)+d,d,1) = T.block(0,i*(d+1)+d,d,1) - t0;
		}

		return T;
	}


	Matrix PGOAgent::getTrajectoryInGlobalFrame(){
		lock_guard<mutex> lock(mPosesMutex);

		Matrix T = globalAnchor.block(0,0,r,d).transpose() * X;
		Matrix t0 = globalAnchor.block(0,0,r,d).transpose() * globalAnchor.block(0,d,r,1);

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
			map[*it] = X.block(0, idx*(d+1), r, d+1);
		}
		return map;
	}

	
	// TODO: return optimization stats 
	void PGOAgent::optimize(){

		if(verbose) cout << "Agent " << mID << " optimize..." << endl;

		// lock pose update
		unique_lock<mutex> tLock(mPosesMutex);

		// need to lock measurements later;
		unique_lock<mutex> mLock(mMeasurementsMutex, std::defer_lock);
		
		// number of poses updated at this time
		unsigned k = n; 


		// read private and shared measurements 
		mLock.lock();
		vector<RelativeSEMeasurement> myMeasurements;
		for(size_t i = 0; i < odometry.size(); ++i){
			RelativeSEMeasurement m = odometry[i];
			if(m.p1 < k && m.p2 < k) myMeasurements.push_back(m);
		}
		for(size_t i = 0; i < privateLoopClosures.size(); ++i){
			RelativeSEMeasurement m = privateLoopClosures[i];
			if(m.p1 < k && m.p2 < k) myMeasurements.push_back(m);
		}
		if (myMeasurements.empty()){
			if (verbose) cout << "No measurements. Skip optimization!" << endl;
			return;
		} 
		vector<RelativeSEMeasurement> sharedMeasurements;
		for(size_t i = 0; i < sharedLoopClosures.size(); ++i){
			RelativeSEMeasurement m = sharedLoopClosures[i];
			assert(m.R.size() != 0);
			assert(m.t.size() != 0);
			if(m.r1 == mID && m.p1 < k) sharedMeasurements.push_back(m);
			else if(m.r2 == mID && m.p2 < k) sharedMeasurements.push_back(m);
		}
		mLock.unlock();


		// construct data matrices
		SparseMatrix Q((d+1)*k, (d+1)*k);
		SparseMatrix G(r,(d+1)*k);
		bool success = constructCostMatrices(myMeasurements, sharedMeasurements, &Q, &G);
		if(!success){
			if (verbose) cout << "Has not received shared pose yet. Skip optimization!" << endl;
			return;
		}


		
		// Read current estimates of the first k poses
		Matrix Xcurr = X.block(0,0,r,(d+1)*k);
		assert(Xcurr.cols() == Q.cols());


		// Construct optimization problem
		QuadraticProblem problem(k, d, r, Q, G);

		// Initialize optimizer object
		QuadraticOptimizer optimizer(&problem);
		optimizer.setVerbose(verbose);
		optimizer.setAlgorithm(algorithm);
		optimizer.setStepsize(stepsize);
		
		// Optimize
		auto startTime = std::chrono::high_resolution_clock::now();
		Matrix Xnext = optimizer.optimize(Xcurr);
		auto counter = std::chrono::high_resolution_clock::now() - startTime;
		double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();
		if(verbose) {
			cout << "Optimization time: " << elapsedMs / 1000 << " seconds." << endl;
			Matrix Xdiff = Xnext - Xcurr;
			cout << "Relative change: " << Xdiff.norm() << endl;
		}

		X.block(0,0,r,(d+1)*k) = Xnext;
		assert(k == n);
	}


	bool PGOAgent::constructCostMatrices(
			const vector<RelativeSEMeasurement>& privateMeasurements,
            const vector<RelativeSEMeasurement>& sharedMeasurements,
            SparseMatrix* Q, 
            SparseMatrix* G)
	{

		// All private measurements appear in the quadratic term
		*Q = constructConnectionLaplacianSE(privateMeasurements);

		// Halt update of shared neighbor poses
		unique_lock<mutex> lock(mNeighborPosesMutex, std::defer_lock);

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
					return false;
				}
				lock.lock();
				Matrix Xj = KVpair->second;
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
				Matrix L = - Xj * Omega * T.transpose();
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
					return false;
				}
				lock.lock();
				Matrix Xi = KVpair->second;
				lock.unlock();

				// Modify quadratic cost
				size_t idx = m.p2;

				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < d+1; ++row){
						Q->coeffRef(idx*(d+1)+row, idx*(d+1)+col) += Omega(row,col);
					}
				}

				// Modify linear cost
				Matrix L = - Xi * T * Omega;
				for (size_t col = 0; col < d+1; ++col){
					for(size_t row = 0; row < r; ++row){
						G->coeffRef(row, idx*(d+1)+col) += L(row,col);
					}
				}

			}

		}

		return true;

	}


	void PGOAgent::startOptimizationLoop(double freq){
		if (isOptimizationRunning()){
			if(verbose) cout << "WARNING: optimization thread already running! Skip..." << endl;
			return;
		}

		rate = freq;	

      	mOptimizationThread = new thread(&PGOAgent::runOptimizationLoop,this);

	}


	void PGOAgent::runOptimizationLoop(){
		if (verbose) cout << "Agent " << mID << " optimization thread running at " << rate << " Hz." << endl;

		// Create exponential distribution with the desired rate
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
	    std::mt19937 rng(rd()); //Standard mersenne_twister_engine seeded with rd()
  		std::exponential_distribution<double> ExponentialDistribution(rate);

		while(true)
		{
			
			double sleepUs = 1e6 * ExponentialDistribution(rng); // sleeping time in microsecond

			usleep(sleepUs);

			optimize();

			// Check if finish requested
			if(mFinishRequested){
				break;
			}

		}
	}

	void PGOAgent::endOptimizationLoop(){
		if(!isOptimizationRunning()) return;

		mFinishRequested = true;

		// wait for thread to finish
		mOptimizationThread->join();
		
		delete mOptimizationThread;
		
		mOptimizationThread = nullptr;

		mFinishRequested = false; // reset request flag

		if (verbose) cout << "Agent " << mID << " optimization thread exited. " << endl;


	}


	bool PGOAgent::isOptimizationRunning(){
		return !(mOptimizationThread == nullptr);
	}


	bool PGOAgent::findSharedLoopClosure(unsigned neighborID, unsigned neighborPose, RelativeSEMeasurement& mOut){

		for(size_t i = 0; i < sharedLoopClosures.size(); ++i){
			RelativeSEMeasurement m = sharedLoopClosures[i];
			if ((m.r1 == neighborID && m.p1 == neighborPose) || (m.r2 == neighborID && m.p2 == neighborPose)){
				mOut = m;
				return true;
			}
		}

		return false;
	}


	


}