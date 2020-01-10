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

    QuadraticProblem* problem = nullptr;

    vector<mutex> mUpdateMutexes;

    vector<vector<unsigned>> adjList;

  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;

    unsigned d,r,n;

    // current iterate
    Matrix Y;
    // number of writes performed by ALL workers
    unsigned numWrites;
    
    // ROPTLIB
    CartanSyncManifold* manifold;
    CartanSyncVariable* x;
    CartanSyncVector* euclideanGradient;
    CartanSyncVector* riemannianGradient;

    
    void initialize();

    float computeCost();

    float computeGradNorm();


  };

} 




#endif