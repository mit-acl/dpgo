#ifndef RGDWORKER_H
#define RGDWORKER_H

#include "multithread/RGDMaster.h"
#include "SESync_utils.h"

/*Define the namespace*/
namespace AsynchPGO{
  
  class RGDMaster;

  class RGDWorker{
  public:
    RGDWorker(RGDMaster* pMaster, unsigned pId);

    void setUpdateIndices(vector<unsigned>& pUpdateIndices){
      updateIndices = pUpdateIndices;
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
    vector<unsigned> updateIndices;
  	RGDMaster* master;
  	bool mFinishRequested;
  	bool mFinished;

  };


} 




#endif