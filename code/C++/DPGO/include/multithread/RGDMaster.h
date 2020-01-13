#ifndef RGDMASTER_H
#define RGDMASTER_H

#include <vector>
#include <thread>
#include <mutex>
#include <Eigen/Dense>
#include "Manifolds/Element.h"
#include "Manifolds/Manifold.h"
#include "QuadraticProblem.h"
#include "multithread/RGDWorker.h"
#include "manifold/LiftedSEManifold.h"
#include "manifold/LiftedSEVariable.h"
#include "manifold/LiftedSEVector.h"

using namespace std;

/*Define the namespace*/
namespace DPGO{

  class RGDWorker;

  class RGDMaster{

  public:
    RGDMaster(QuadraticProblem* p, Matrix Y0);

    ~RGDMaster();

    void solve(unsigned num_threads);
    
    void readComponent(unsigned i, Matrix& Yi);

    void writeComponent(unsigned i, Matrix& Yi);

    void readDataMatrixBlock(unsigned i, unsigned j, Matrix& Qij);

    void getSolution(Matrix& Yout){
      Yout = Y;
    }

    QuadraticProblem* problem = nullptr;

    vector<mutex> mUpdateMutexes;

    vector<vector<unsigned>> adjList;

    // number of writes performed by ALL workers
    unsigned numWrites;

  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;

    unsigned d,r,n;

    // current iterate
    Matrix Y;
    
    // ROPTLIB
    LiftedSEManifold* M;
    LiftedSEVariable* Var;
    LiftedSEVector* EGrad;
    LiftedSEVector* RGrad;

    
    void initialize();

    float computeCost();

    float computeGradNorm();


  };

} 




#endif