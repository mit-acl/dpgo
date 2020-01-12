#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include "ProductManifold.h"

/*Define the namespace*/
namespace DPGO{

  class CartanManifold : public ROPTLIB::ProductManifold{
  public:
    CartanManifold(integer r, integer d);

    virtual ~CartanManifold();

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


#endif