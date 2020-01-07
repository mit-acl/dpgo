#include "multithread/RGDMaster.h"
#include <iostream>
#include <cassert>
#include <unistd.h>

using namespace std;

namespace AsynchPGO{

	RGDMaster::RGDMaster(QuadraticProblem* p){
		problem = p;
		initialize();
	}

	void RGDMaster::initialize(){
		assert(problem != nullptr);
		vector<mutex> list(problem->num_poses());
		mUpdateMutexes.swap(list);		
	}

	void RGDMaster::solve(int num_threads){
		assert(num_threads > 0);

		count = 0;
		
		for(unsigned i = 0; i < (unsigned) num_threads; ++i){
			// initialize a new worker
			RGDWorker* worker = new RGDWorker(this, i);
			workers.push_back(worker);

			// initialize thread that this worker runs on
			thread* worker_thread = new thread(&AsynchPGO::RGDWorker::run, worker);
			threads.push_back(worker_thread);
		}

		while(true){
			// cout << "Count: " << count << endl;

			if (count > 1000){
				// stop all workers
				for(unsigned i = 0; i < workers.size(); ++i){
					workers[i]->requestFinish();
				}
				break;
			}

			usleep(5000);
		}

		// pause until all workers finish
		for(unsigned i = 0; i < threads.size(); ++i){
			threads[i]->join();
		}

		cout << "Master finished." << endl;

	}

	void RGDMaster::increment(){
		count++;
	}
}