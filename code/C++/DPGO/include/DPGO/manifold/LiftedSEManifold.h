/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef LIFTEDSEMANIFOLD_H
#define LIFTEDSEMANIFOLD_H

#include "Manifolds/ProductManifold.h"
#include "Manifolds/Stiefel/Stiefel.h"
#include "Manifolds/Euclidean/Euclidean.h"

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