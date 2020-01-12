#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include "ProductManifold.h"
#include "Stiefel.h"
#include "Euclidean.h"

/*Define the namespace*/
namespace DPGO{

  class LiftedSEManifold{
  public:
    LiftedSEManifold(int r, int d, int n);

    ~LiftedSEManifold();

    ROPTLIB::ProductManifold* getManifold(){
      return MyManifold;
    }
  
  private:
    ROPTLIB::Stiefel* StiefelManifold;
    ROPTLIB::Euclidean* EuclideanManifold;
    ROPTLIB::ProductManifold* CartanManifold;
    ROPTLIB::ProductManifold* MyManifold;

  };
} /*end of SESync namespace*/


#endif