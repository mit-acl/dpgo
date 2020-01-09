#include "multithread/RGDMaster.h"
#include <iostream>
#include <cassert>
#include <unistd.h>

using namespace std;

namespace AsynchPGO{

	RGDMaster::RGDMaster(QuadraticProblem* p, Matrix Y0){
		problem = p;
		Y = Y0;
		initialize();
	}

	void RGDMaster::initialize(){
		assert(problem != nullptr);
		unsigned n = problem->num_poses();
		unsigned d = problem->dimension();

		// create mutexes
		vector<mutex> list(n);
		mUpdateMutexes.swap(list);		


		// compute adjacency list
		for(unsigned i = 0; i < n; ++i){
			vector<unsigned> empty_list;
			adjList.push_back(empty_list);
			for(unsigned j = 0; j < n; ++j){
				if (i == j) continue;
				unsigned rowStart = (d+1) * i;
				unsigned colStart = (d+1) * j;
				if(problem->Q.block(rowStart, colStart, d+1, d+1).norm() > 0.0001){
					adjList[i].push_back(j);
				}
			}
		}

	}

	void RGDMaster::solve(unsigned num_threads){

		if(num_threads == 0){
			cout << "At least one worker must be used. " << endl;
			return;
		}

		numWrites = 0;

		unsigned n = problem->num_poses();
		unsigned numPosesPerWorker = n / num_threads;
		assert(numPosesPerWorker != 0);
		if(numPosesPerWorker == 0){
			cout << "Idle workers detected. Try decrease the number of workers." << endl;
			return;
		}

		// compute initial cost
		float initialCost = (Y * problem->Q * Y.transpose()).trace();
		cout << "Initial cost:  " << initialCost << endl;

		for(unsigned i = 0; i < num_threads; ++i){
			// initialize a new worker
			RGDWorker* worker = new RGDWorker(this, i);
			workers.push_back(worker);

			// compute the poses that this worker updates
			vector<unsigned> updateIndices;
			unsigned indexStart = numPosesPerWorker * i;
			unsigned indexEnd = numPosesPerWorker * (i+1) - 1;
			if(i == num_threads - 1){
				indexEnd = n-1;
			}
			for(unsigned idx = indexStart; idx <= indexEnd; ++idx){
				updateIndices.push_back(idx);
			}
			worker->setUpdateIndices(updateIndices);

			// initialize thread that this worker runs on
			thread* worker_thread = new thread(&AsynchPGO::RGDWorker::run, worker);
			threads.push_back(worker_thread);
		}

		sleep(5);

		while(false){

			float cost = (Y * problem->Q * Y.transpose()).trace();
			cout << "Cost = " << cost << endl;

			if (true){
				// stop all workers
				for(unsigned i = 0; i < workers.size(); ++i){
					workers[i]->requestFinish();
				}
				break;
			}

			usleep(50000);
		}

		// pause until all workers finish
		for(unsigned i = 0; i < threads.size(); ++i){
			threads[i]->join();
		}

		float finalCost = (Y * problem->Q * Y.transpose()).trace();

		cout << "Master finished." << endl;
		cout << "Initial cost:  " << initialCost << ". Final cost: " << finalCost << ". Number of writes: " << numWrites << "." << endl;

	}

	void RGDMaster::readComponent(unsigned i, Matrix& Yi){
		unsigned d = problem->dimension();
		unsigned r = problem->relaxation_rank();

		unsigned start = (d+1) * i;

		Yi = Y.block(0, start, r, d+1);
	}

    void RGDMaster::writeComponent(unsigned i, Matrix& Yi){
    	unsigned d = problem->dimension();
		unsigned r = problem->relaxation_rank();

		unsigned start = (d+1) * i;

		Y.block(0, start, r, d+1) = Yi;
		numWrites++;
    }

    void RGDMaster::readDataMatrixBlock(unsigned i, unsigned j, Matrix& Qij){
    	unsigned d = problem->dimension();
    	unsigned rowStart = (d+1) * i;
    	unsigned colStart = (d+1) * j;

    	Qij = Matrix(problem->Q.block(rowStart, colStart, d+1, d+1));
    }
}