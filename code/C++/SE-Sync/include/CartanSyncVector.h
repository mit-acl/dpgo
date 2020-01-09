/*
This file defines the class of a point on the manifold defined for Cartan-Sync.

SmartSpace --> ProductElement --> CartanSyncVector

* jbriales 03 Sept 2017
*/

#ifndef CARTANSYNCVECTOR_H
#define CARTANSYNCVECTOR_H

#include "Manifolds/ProductElement.h"
#include "SESync_types.h"

/*Define the namespace*/
namespace SESync{

  class CartanVector : public ROPTLIB::ProductElement{
  public:
    CartanVector(integer r, integer d);
  };

  class CartanSyncVector : public ROPTLIB::ProductElement{
  public:
    CartanSyncVector(integer r, integer d, integer n);

    virtual ~CartanSyncVector();

    CartanSyncVector* ConstructEmpty(void) const;

  public:
    /** The maximum rank of the lifted domain */
    unsigned int r = 0;

    /** Dimensionality of the Euclidean space */
    unsigned int d = 0;

    /** Number of poses */
    unsigned int n = 0;

  };
} /*end of SESync namespace*/
#endif // end of CartanSyncVector_H
