
#include "CartanSyncManifold.h"

#include "CartanSyncVariable.h" // necessary for setting EMPTYEXTR
//#include "CartanSyncVector.h"

#include "Stiefel.h"
#include "Euclidean.h"

namespace SESync{

  CartanManifold::CartanManifold(integer r, integer d) :
    ProductManifold(2,
                    new ROPTLIB::Stiefel(r, d),1,
                    new ROPTLIB::Euclidean(r),1)
  {
    // Set properties for internal manifold
    ROPTLIB::Stiefel* StiefelMani =
        static_cast<ROPTLIB::Stiefel*>( this->GetManifold(0) );
    StiefelMani->ChooseStieParamsSet3();
    // Use the Euclidean metric, QF retraction, vector transport by projection and
    // extrinsic representation
    // StiefelMani->ChooseStieParamsSet4();
  }

  CartanSyncManifold::CartanSyncManifold(integer r, integer d, integer n) :
    ProductManifold(1, new CartanManifold(r,d), n)
  {
    name.assign("Product of lifted SE(d)'s");
    this->SetIsIntrApproach(false);

    // EMPTYEXTR should be already allocated by the ProductElement constructor,
    // but we want it to be a CartanSyncVariable for convenient interfacing
    // TODO: Define CartanSyncElement rather than variable,
    // as a common basis for Variable and Vector types?
    delete EMPTYEXTR;
    EMPTYEXTR = new CartanSyncVariable(r,d,n);

    // We do not set EMPTYINTR as the INTR approach is not used in our code
  }



  // Use default?
  CartanSyncManifold::~CartanSyncManifold(void)
  { }

} /*end of SESync namespace*/
