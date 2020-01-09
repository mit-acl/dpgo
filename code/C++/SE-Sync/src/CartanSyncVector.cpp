
#include "CartanSyncVector.h"

// TODO: Remove dependencies on Cartan2... methods here
#include "SESync_utils.h"

#include "SESync_types.h"

// For the ROPTLIB types stored internally
#include "Stiefel.h"
#include "Euclidean.h"
  
namespace SESync{

  CartanVector::CartanVector(integer r, integer d) :
    ProductElement(2,
                    new ROPTLIB::StieVector(r, d),1,
                    new ROPTLIB::EucVector(r),1)
  { }

  CartanSyncVector::CartanSyncVector(integer r, integer d, integer n) :
    ProductElement(1, new CartanVector(r,d), n), r(r), d(d), n(n)
  {
  }

  CartanSyncVector::~CartanSyncVector(void)
  {
  }

  CartanSyncVector *CartanSyncVector::ConstructEmpty() const
  {
    return new CartanSyncVector(r,d,n);
  }

} /*end of SESync namespace*/
