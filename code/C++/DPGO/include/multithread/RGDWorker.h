#ifndef RGDWORKER_H
#define RGDWORKER_H

#include "multithread/RGDMaster.h"
#include "manifold/LiftedSEManifold.h"
#include "manifold/LiftedSEVariable.h"
#include "manifold/LiftedSEVector.h"


/*Define the namespace*/
namespace DPGO{
  
  class RGDMaster;

  class RGDWorker{
  public:
    RGDWorker(RGDMaster* pMaster, unsigned pId);

    ~RGDWorker();

    void setUpdateIndices(vector<unsigned>& pUpdateIndices){
      updateIndices = pUpdateIndices;
    }

    void setUpdateRate(int freq){
      double sleepSec = 1 / (float) freq;
      sleepMicroSec = (int) (sleepSec * 1000 * 1000);
    }

    void setStepsize(float s){
      stepsize = s;
    }

    void run();

    void requestFinish();

    bool isFinished();
  
  private:

    void readComponent(unsigned i, Matrix& Yi);

    void writeComponent(unsigned i, Matrix& Yi);

    void computeEuclideanGradient(unsigned i, Matrix& Gi);

    void gradientUpdate(Matrix& Yi, Matrix& Gi, Matrix& YiNext);

  	RGDMaster* master;
    unsigned id;
    bool mFinishRequested;
    bool mFinished;

    int sleepMicroSec;
    float stepsize;

    unsigned d, r;
    vector<unsigned> updateIndices;


    // ROPTLIB
    LiftedSEManifold* M;
    LiftedSEVariable* Var;
    LiftedSEVariable* VarNext;
    LiftedSEVector* EGrad;
    LiftedSEVector* RGrad;
    LiftedSEVector* Eta;

  };


} 




#endif