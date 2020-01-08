#ifndef RGDWORKER_H
#define RGDWORKER_H

#include "multithread/RGDMaster.h"

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

    void increment();

    void requestFinish();

    bool isFinished();
  
  private:

  	unsigned id;
    vector<unsigned> updateIndices;
  	RGDMaster* master;
  	bool mFinishRequested;
  	bool mFinished;

  };


} 




#endif