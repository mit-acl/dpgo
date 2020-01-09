#ifndef RGDMASTER_H
#define RGDMASTER_H

#include <vector>
#include <thread>
#include <mutex>
#include <Eigen/Dense>
#include "QuadraticProblem.h"
#include "CartanSyncVariable.h"
#include "CartanSyncManifold.h"
#include "CartanSyncVector.h"
#include "multithread/RGDWorker.h"

using namespace std;

/*Define the namespace*/
namespace AsynchPGO{

  class RGDWorker;

  class RGDMaster{

  public:
    RGDMaster(QuadraticProblem* p, Matrix Y0);

    void solve(unsigned num_threads);
    
    void readComponent(unsigned i, Matrix& Yi);

    void writeComponent(unsigned i, Matrix& Yi);

    void readDataMatrixBlock(unsigned i, unsigned j, Matrix& Qij);

    vector<mutex> mUpdateMutexes;

    vector<vector<unsigned>> adjList;

  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;
    QuadraticProblem* problem = nullptr;


    // current iterate
    Matrix Y;

    unsigned numWrites;

    void initialize();

  };

} 




#endif