/*
This file defines the class of a point on the manifold defined for Cartan-Sync.

SmartSpace --> ProductElement --> CartanSyncVariable

* jbriales 03 Sept 2017
*/

#ifndef CARTANSYNCVARIABLE_H
#define CARTANSYNCVARIABLE_H

#include "Manifolds/ProductElement.h"
#include "SESync_types.h"

/*Define the namespace*/
namespace SESync{

  class CartanVariable : public ROPTLIB::ProductElement{
  public:
    CartanVariable(integer r, integer d);
  };

  class CartanSyncVariable : public ROPTLIB::ProductElement{
  public:
    CartanSyncVariable(integer r, integer d, integer n);

    virtual ~CartanSyncVariable();

    CartanSyncVariable* ConstructEmpty(void) const;

    /*Construct new variable augmenting the r dimension in the input variable */
    void lift(CartanSyncVariable& Xlift) const;

    /*Project augmented variable to smaller domain */
    void unlift(CartanSyncVariable& Xunlift) const;

    /*Print this point on the manifold as a ...*/
//    virtual void Print(const char *name = "", bool isonlymain = true) const;


    /// ACCESSORS
    /// Convenient methods to access different data blocks in the variable

    /* Return whole variable matrix as [R_1,t_1,...,R_n,t_n] */
    Matrix mat(void);
    /* Same as above, read-only (const) */
    Matrix mat(void) const;

    /* Return row block-vector of rotations [R_1,...,R_n] */
    Matrix R(void);

    /* Return i-th rotation R_i */
    Matrix R(size_t i);

    /* Return row block-vector of translations [t_1,...,t_n] */
    Matrix t(void);

    /* Return i-th translation t_i */
    Matrix t(size_t i);

  public:
//  protected:
    /** The maximum rank of the lifted domain */
    unsigned int r = 0;

    /** Dimensionality of the Euclidean space */
    unsigned int d = 0;

    /** Number of poses */
    unsigned int n = 0;

    /** Eigen matrix pointing to the same data array as the ROPTLIB variable */
//    Matrix Y;
  };
} /*end of SESync namespace*/
#endif // end of CARTANSYNCVARIABLE_H
