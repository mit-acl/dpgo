
#include "CartanSyncVector.h"
#include "SESync_utils.h"
#include "SESync_types.h"
  
namespace SESync{

  CartanVector::CartanVector(integer r, integer d) :
    ProductElement(2,
                    new ROPTLIB::StieVector(r, d),1,
                    new ROPTLIB::EucVector(r),1)
  {}

  CartanVector::~CartanVector(void)
  {
  }

  CartanSyncVector::CartanSyncVector(integer r, integer d, integer n) :
    ProductElement(1, new CartanVector(r,d), n)
  {
  }

  CartanSyncVector::~CartanSyncVector(void)
  {
  }

  // CartanSyncVector *CartanSyncVector::ConstructEmpty() const
  // {
  //   return new CartanSyncVector(r,d,n);
  // }

} /*end of SESync namespace*/
