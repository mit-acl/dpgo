#ifndef SOLVERMASTER_H
#define SOLVERMASTER_H

#include <vector>
#include <thread>
#include <mutex>
#include <Eigen/Dense>
#include "QuadraticProblem.h"
#include "multithread/RGDWorker.h"

using namespace std;

/*Define the namespace*/
namespace AsynchPGO{

  class RGDWorker;

  class RGDMaster{

  public:
    RGDMaster(QuadraticProblem* p);

    void solve(int num_threads);

    // tutorial
    void increment();

    
    vector<mutex> mUpdateMutexes;

    


  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;

    QuadraticProblem* problem = nullptr;

    void initialize();
  	
    // tutorial
  	int count;

  };

} 




#endif