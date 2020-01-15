#ifndef LIFTEDSEVECTOR_H
#define LIFTEDSEVECTOR_H

#include "ProductManifold.h"
#include "Stiefel.h"
#include "Euclidean.h"
#include "DPGO_types.h"

/*Define the namespace*/
namespace DPGO{

  class LiftedSEVector{
  public:
    LiftedSEVector(int r, int d, int n);

    ~LiftedSEVector();

    ROPTLIB::ProductElement* vec(){
      return MyVector;
    }

    void getData(Matrix& Y);

    void setData(const Matrix& Y);

  private:
  	ROPTLIB::StieVector* StiefelVector;
  	ROPTLIB::EucVector* EuclideanVector;
  	ROPTLIB::ProductElement* CartanVector;
  	ROPTLIB::ProductElement* MyVector;
  };
} /*end of SESync namespace*/


#endif