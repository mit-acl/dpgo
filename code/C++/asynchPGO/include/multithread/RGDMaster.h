#ifndef SOLVERMASTER_H
#define SOLVERMASTER_H

#include <vector>
#include <thread>
#include <Eigen/Dense>
#include "QuadraticProblem.h"
#include "multithread/RGDWorker.h"

using namespace std;

/*Define the namespace*/
namespace AsynchPGO{

  class RGDWorker;

  class RGDMaster{

  public:
    RGDMaster(){}

    void setProblem(QuadraticProblem* p){
      problem = p;
    }

    void solve(int num_threads);

    // tutorial
    void increment();


  private:
  	vector<thread*> threads;
  	vector<RGDWorker*> workers;

    QuadraticProblem* problem = nullptr;
  	
    // tutorial
  	int count;

  };

} 




#endif