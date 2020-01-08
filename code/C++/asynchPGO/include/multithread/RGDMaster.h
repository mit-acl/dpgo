#ifndef RGDMASTER_H
#define RGDMASTER_H

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
    RGDMaster(QuadraticProblem* p, Matrix Y0);

    void solve(unsigned num_threads);
    
    vector<mutex> mUpdateMutexes;

  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;
    QuadraticProblem* problem = nullptr;
    vector<vector<unsigned>> adjList;


    // current iterate
    Matrix Y;

    void initialize();

  };

} 




#endif