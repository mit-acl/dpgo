#include <iostream>
#include <chrono>
#include <random>
#include <unistd.h>
#include "multithread/RGDWorker.h"

using namespace std;


namespace DPGO{

	RGDWorker::RGDWorker(RGDMaster* pMaster, unsigned pId):
	master(pMaster), 
	id(pId), 
	mFinishRequested(false), 
	mFinished(false),
	sleepMicroSec(5000), //default rate is 200 Hz
	stepsize(0.001)
	{

		d = master->dimension();
		r = master->relaxation_rank();

		M = new LiftedSEManifold(r,d,1);
		Var = new LiftedSEVariable(r,d,1);
		VarNext = new LiftedSEVariable(r,d,1);
		EGrad = new LiftedSEVector(r,d,1);
		RGrad = new LiftedSEVector(r,d,1);
		Eta = new LiftedSEVector(r,d,1);

		cout << "Worker " << id << " initialized. "<< endl;
	}

	RGDWorker::~RGDWorker(){
		delete M;
		delete Var;
		delete VarNext;
		delete EGrad;
		delete RGrad;
		delete Eta;
	}

	void RGDWorker::run(){
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
	    std::mt19937 rng(rd()); //Standard mersenne_twister_engine seeded with rd()
	    std::uniform_int_distribution<> distribution(0, updateIndices.size()-1);


	    auto startTime = std::chrono::high_resolution_clock::now();
	    double numWrites = 0.0;


		while(true){

			// randomly select an index
			unsigned i = updateIndices[distribution(rng)];

			Matrix Yi, Gi, YiNext;
			readComponent(i, Yi);

			Gi.resize(Yi.rows(), Yi.cols());
			computeEuclideanGradient(i, Gi);

			gradientUpdate(Yi, Gi, YiNext);

			writeComponent(i, YiNext);

			numWrites+=1.0;

			if(mFinishRequested) break;
			
			// use usleep for microsecond
			usleep(sleepMicroSec); 
		}

		auto counter = std::chrono::high_resolution_clock::now() - startTime;
		double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();

		mFinished = true;
		cout << "Worker " << id << " finished. Average runtime per iteration (sleep included): " \
			<< elapsedMs/numWrites  << " milliseconds." << endl;
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
    	Var->setData(Yi);
    	EGrad->setData(Gi);
    	M->getManifold()->Projection(Var->var(), EGrad->vec(), RGrad->vec());
    	M->getManifold()->ScaleTimesVector(Var->var(), -stepsize, RGrad->vec(), Eta->vec());
    	M->getManifold()->Retraction(Var->var(), Eta->vec(), VarNext->var());
    	VarNext->getData(YiNext);
    }
}