/*
* This file defines the class for the manifold
* M(r,d,n) = \{X \in stack({St(r,d) x R^r}^n \},
* which characterizes the domain for the Cartan-Sync algorithm.
* This is a product of n "lifted poses", where a lifted pose is in turn the
* product of a Stiefel and a linear manifolds.
*
* jbriales 03 Sept 2017
*/

#ifndef CARTANSYNCMANIFOLD_H
#define CARTANSYNCMANIFOLD_H

#include "ProductManifold.h"

/*Define the namespace*/
namespace SESync{

  class CartanManifold : public ROPTLIB::ProductManifold{
  public:
    CartanManifold(integer r, integer d);
  };

  class CartanSyncManifold : public ROPTLIB::ProductManifold{
  public:
    CartanSyncManifold() = default; // explicitly defaulted

    /*Construct an CartanSyncManifold manifold, which is a product of lifted
     * poses. The number of lifted poses is n, and each lifted pose is a
     * Cartan group of a r x d Stiefel manifold and a linear space in R^r.*/
    CartanSyncManifold(integer r, integer d, integer n);

    /*Delete each component manifold*/
    virtual ~CartanSyncManifold();
  };
} /*end of SESync namespace*/
#endif // end of CARTANSYNCMANIFOLD_H
