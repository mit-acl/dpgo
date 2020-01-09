#include <iostream>
#include <unistd.h>
#include <random>
#include <Eigen/SVD>
#include "multithread/RGDWorker.h"


using namespace std;

namespace AsynchPGO{

	RGDWorker::RGDWorker(RGDMaster* pMaster, unsigned pId){
		id = pId;
		master = pMaster;
		mFinishRequested = false;
		mFinished = false;

		cout << "Worker " << id << " initialized. "<< endl;
	}

	void RGDWorker::run(){
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
	    std::mt19937 rng(rd()); //Standard mersenne_twister_engine seeded with rd()
	    std::uniform_int_distribution<> distribution(0, updateIndices.size()-1);

		while(true){

			// randomly select an index
			unsigned i = updateIndices[distribution(rng)];

			Matrix Yi, Gi, YiNext;
			readComponent(i, Yi);

			Gi.resize(Yi.rows(), Yi.cols());
			computeEuclideanGradient(i, Gi);

			gradientUpdate(Yi, Gi, YiNext);

			writeComponent(i, YiNext);

			if(mFinishRequested) break;
			
			// use usleep for microsecond
		sleep(1); 
		}

		mFinished = true;
		cout << "Worker " << id << " finished." << endl;
	}

	void RGDWorker::requestFinish(){
		mFinishRequested = true;
	}

	bool RGDWorker::isFinished(){
		return mFinished;
	}

	void RGDWorker::readComponent(unsigned i, Matrix& Yi){
		// obtain lock
		lock_guard<mutex> lock(master->mUpdateMutexes[i]);
		master->readComponent(i, Yi);
	}

    void RGDWorker::writeComponent(unsigned i, Matrix& Yi){
    	// obtain lock
		lock_guard<mutex> lock(master->mUpdateMutexes[i]);
		master->writeComponent(i, Yi);
    }

    void RGDWorker::computeEuclideanGradient(unsigned i, Matrix &Gi){
    	Gi.setZero();
    	// iterate over neighbors of i
    	for(unsigned k = 0; k < master->adjList[i].size(); ++k){
    		unsigned j = master->adjList[i][k];
    		Matrix Yj, Qji;
    		master->readComponent(j, Yj);
    		master->readDataMatrixBlock(j, i, Qji);
    		Gi = Gi + Yj * Qji;
    	}
    }

    void RGDWorker::gradientUpdate(Matrix& Yi, Matrix& Gi, Matrix& YiNext){
    	YiNext.setZero();
    	unsigned r = Yi.rows();
    	unsigned d = Yi.cols() - 1;
    	CartanSyncManifold manifold(r,d,1);
    	CartanSyncVariable x(r,d,1);
    	CartanSyncVector euclideanGradient(r,d,1);
    	CartanSyncVector riemannianGradient(r,d,1);
    	Mat2CartanProd(Yi, x);
    	Mat2CartanProd(Gi, euclideanGradient);
    	
    	// Compute Riemannian gradient
    	manifold.Projection(&x, &euclideanGradient, &riemannianGradient);

    	Matrix RG;
    	CartanProd2Mat(riemannianGradient, RG);
    	cout << RG.norm() << endl;

    	YiNext = Yi;

    	// Compute descent direction
    	// CartanSyncVector eta(r,d,1);
    	// manifold.ScaleTimesVector(&x, -0.0000001, &riemannianGradient, &eta);

    	// Update
    	// CartanSyncVariable xNext(r,d,1);
    	// manifold.Retraction(&x, &eta, &xNext);

    	// CartanProd2Mat(xNext, YiNext);
    }
}