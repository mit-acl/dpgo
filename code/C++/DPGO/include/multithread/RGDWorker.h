#ifndef RGDWORKER_H
#define RGDWORKER_H

#include "multithread/RGDMaster.h"
#include "SESync_utils.h"
#include "CartanSyncVariable.h"
#include "CartanSyncManifold.h"
#include "CartanSyncVector.h"
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

  	unsigned id;
    unsigned d, r;
    vector<unsigned> updateIndices;
    int sleepMicroSec = 5000; // default update rate = 200 Hz
    float stepsize = 0.001;
  	RGDMaster* master;
  	bool mFinishRequested;
  	bool mFinished;

    // ROPTLIB
    CartanSyncManifold* manifold;
    CartanSyncVariable* x;
    CartanSyncVariable* xNext;
    CartanSyncVector* euclideanGradient;
    CartanSyncVector* riemannianGradient;
    CartanSyncVector* descentVector;

  };


} 




#endif