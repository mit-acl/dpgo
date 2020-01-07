#ifndef SOLVERWORKER_H
#define SOLVERWORKER_H

#include "multithread/RGDMaster.h"

/*Define the namespace*/
namespace AsynchPGO{
  
  class RGDMaster;

  class RGDWorker{
  public:
    RGDWorker(RGDMaster* pMaster, unsigned pId);

    void run();

    void requestFinish();

    bool isFinished();
  
  private:

  	unsigned id;
  	RGDMaster* master;
  	bool mFinishRequested;
  	bool mFinished;

  };


} 




#endif