/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include "distributed/PGOAgent.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <random>
#include "DPGO_utils.h"
#include "QuadraticProblem.h"
#include "QuadraticOptimizer.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>

using namespace std;

namespace DPGO{

	PGOAgent::PGOAgent(unsigned ID, const PGOAgentParameters& params): 
	mID(ID), 
	mCluster(0), 
	d(params.d), 
	r(params.r), 
	n(1),
	verbose(params.verbose),
	rate(1),
	algorithm(params.algorithm),
	stepsize(1e-3)
	{
		// automatically initialize the first pose on the Cartan group
		LiftedSEVariable x(r,d,1);
		x.var()->RandInManifold();
		Y = x.getData();

		// initialize globalAnchor
		globalAnchor = Y;
	}


	PGOAgent::~PGOAgent(){
		// Make sure that optimization thread is not running, before exiting
		endOptimizationLoop();
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
		
		Matrix Y_ = Y;
		assert(Y_.cols() == (d+1)*n);
		assert(Y_.rows() == r);
		Y = Matrix::Zero(r,(d+1)*(n+1));
		Y.block(0,0,r,(d+1)*n) = Y_;

		Matrix currR = Y.block(0, (n-1)*(d+1), r, d);
		Matrix currt = Y.block(0, (n-1)*(d+1)+d,r,1);

		// initialize next pose by propagating odometry
		Matrix nextR = currR * factor.R;
		Matrix nextt = currt + currR * factor.t;
		Y.block(0, n*(d+1), r, d) = nextR;
		Y.block(0, n*(d+1)+d,r,1) = nextt;

		n++;
		assert((d+1)*n == Y.cols());
		if(verbose) cout << "Agent " << mID << " initialized pose " << n << endl;

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
			cout << "Agent " << mID << " informed by agent " << neighborID << " to join cluster " << neighborCluster << "!" << endl;
			if (r != d){
				cout << "Error: cluster merging only supports r = d!" << endl;
				assert(r == d);
			}

			// Halt pose update
			if(verbose) cout << "Agent " << mID << " halt optimization thread..." << endl;
			endOptimizationLoop();

			// Halt insertion of new poses
			lock_guard<mutex> tLock(mPosesMutex);
			assert(Y.cols() == n*(d+1));

			// Halt insertion of new measurements
			lock_guard<mutex> mLock(mMeasurementsMutex);

			// Clear cache
			lock_guard<mutex> nLock(mNeighborPosesMutex);
			neighborPoseDict.clear();

			mCluster = neighborCluster;

			// Find the corresponding inter-robot loop closure
			RelativeSEMeasurement m;
			assert(findSharedLoopClosure(neighborID, neighborPose, m));

			// Form relative transformation matrix in homogeneous form
			Matrix dT = Matrix::Identity(d+1, d+1);
			dT.block(0,0,d,d) = m.R;
			dT.block(0,d,d,1) = m.t;

			// Initialize matrices used later
			Matrix Xstar, Xcurr;

			if(m.r1 == neighborID){
				// Incoming edge
				Xstar = var * dT; 
				Xcurr = Y.block(0, m.p2*(d+1), d, d+1);
			}
			else{
				// Outgoing edge
				Xstar = var * dT.inverse();
				Xcurr = Y.block(0, m.p1*(d+1), d, d+1);
			}


			// Desired pose for index pose
			Matrix Tstar = Matrix::Identity(d+1,d+1);
			Tstar.block(0,0,d,d+1) = Xstar;

			// Current pose for index pose
			Matrix Tcurr = Matrix::Identity(d+1,d+1);
			Tcurr.block(0,0,d,d+1) = Xcurr;

			// Required transformation
			Matrix Tc = Tstar * Tcurr.inverse();

			Matrix T1 = Matrix::Identity(d+1, d+1);
			for(size_t i = 0; i < n; ++i){
				T1.block(0,0,d,d+1) = Y.block(0, i*(d+1), d, d+1);
				Matrix T2 = Tc * T1;
				Y.block(0, i*(d+1), d, d+1) = T2.block(0,0,d,d+1);
			}
			
			startOptimizationLoop(rate);
		}

		// Do not store this pose if it comes from a different cluster
		if(neighborCluster != mCluster) return;

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


	Matrix PGOAgent::getTrajectoryInGlobalFrame(){
		lock_guard<mutex> lock(mPosesMutex);

		Matrix T = globalAnchor.block(0,0,r,d).transpose() * Y;
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
			map[*it] = Y.block(0, idx*(d+1), r, d+1);
		}
		return map;
	}

	

	void PGOAgent::optimize(){
		if(verbose) cout << "Agent " << mID << " optimize..." << endl;

		// need to lock pose later
		unique_lock<mutex> tLock(mPosesMutex, std::defer_lock);

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
			if(m.p1 < k && m.p2 < k) myMeasurements.push_back(robustifyMeasurement(m));
		}
		if (myMeasurements.empty()){
			if (verbose) cout << "No measurements. Skip optimization." << endl;
			return;
		} 
		vector<RelativeSEMeasurement> sharedMeasurements;
		for(size_t i = 0; i < sharedLoopClosures.size(); ++i){
			RelativeSEMeasurement m = sharedLoopClosures[i];
			assert(m.R.size() != 0);
			assert(m.t.size() != 0);
			if(m.r1 == mID && m.p1 < k) sharedMeasurements.push_back(robustifyMeasurement(m));
			else if(m.r2 == mID && m.p2 < k) sharedMeasurements.push_back(robustifyMeasurement(m));
		}
		mLock.unlock();


		// construct data matrices
		SparseMatrix Q((d+1)*k, (d+1)*k);
		SparseMatrix G(r,(d+1)*k);
		constructCostMatrices(myMeasurements, sharedMeasurements, &Q, &G);


		
		// Read current estimates of the first k poses
		tLock.lock();
		Matrix Ycurr = Y.block(0,0,r,(d+1)*k);
		assert(Ycurr.cols() == Q.cols());
		tLock.unlock();


		// Construct optimization problem
		QuadraticProblem problem(k, d, r, Q, G);

		// Initialize optimizer object
		QuadraticOptimizer optimizer(&problem);
		optimizer.setVerbose(verbose);
		optimizer.setAlgorithm(algorithm);
		optimizer.setStepsize(stepsize);
		
		// Optimize
		auto startTime = std::chrono::high_resolution_clock::now();
		Matrix Ynext = optimizer.optimize(Ycurr);
		auto counter = std::chrono::high_resolution_clock::now() - startTime;
		double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();
		if(verbose) cout << "Optimization time: " << elapsedMs / 1000 << " seconds." << endl;
		gradnorm = problem.gradNorm(Ynext);

		tLock.lock();
		Y.block(0,0,r,(d+1)*k) = Ynext;
		assert(n == k);
	}


	void PGOAgent::constructCostMatrices(
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
					if(verbose) cout << "WARNING: shared pose does not exist!" << endl;
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
					if(verbose) cout << "WARNING: shared pose does not exist!" << endl;
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

			if(verbose) cout << "Agent " << mID << " optimization thread: sleep for " << sleepUs / 1e6 << " sec..." << endl;

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


	RelativeSEMeasurement PGOAgent::robustifyMeasurement(const RelativeSEMeasurement& m){
		RelativeSEMeasurement mOut = m;
		

		// form the relative SE(d) transformation in homogeneous form 
		Matrix Tij = Matrix::Zero(d+1,d+1);
		Tij.block(0,0,d,d) = m.R;
		Tij.block(0,d,d,1) = m.t;
		Tij(d,d) = 1;


		// form the diagonal weight matrix 
		Matrix Omega = Matrix::Zero(d+1,d+1);
		for(size_t i = 0; i < d; ++i) Omega(i,i) = m.kappa;
		Omega(d,d) = m.tau;


		// retrieve involved variables
		Matrix Yi, Yj;
		if (m.r1 == mID && m.r2 == mID){
			// private factor
			Yi = getYComponent(m.p1);
			Yj = getYComponent(m.p2); 
		}
		else if (m.r1 != mID && m.r2 != mID){
			// discard
			if(verbose) cout << "WARNING: robustified measurement does not belong to this robot! " << endl;
		}
		else{
			// shared factor
			if (m.r1 == mID){
				Yi = getYComponent(m.p1);
				// neighbor ID
				const PoseID nID = make_pair(m.r2, m.p2);
				auto KVpair = neighborPoseDict.find(nID);
				if(KVpair == neighborPoseDict.end()){
					if(verbose) cout << "WARNING: shared pose does not exist!" << endl;
				}
				Yj = KVpair->second;

			}else{
				Yj = getYComponent(m.p2);
				// neighbor ID
				const PoseID nID = make_pair(m.r1, m.p1);
				auto KVpair = neighborPoseDict.find(nID);
				if(KVpair == neighborPoseDict.end()){
					if(verbose) cout << "WARNING: shared pose does not exist!" << endl;
				}
				Yi = KVpair->second;
			}
		}


		// compute scalar residual
		Matrix Yerror = Yj - Yi * Tij;
		double residual = (Yerror * Omega * Yerror.transpose()).trace();
		cout << "Residual = " << residual << endl;

		return mOut;
	}


}