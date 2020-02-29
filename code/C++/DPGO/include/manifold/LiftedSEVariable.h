#ifndef LIFTEDSEVARIABLE_H
#define LIFTEDSEVARIABLE_H

#include "Manifolds/ProductElement.h"
#include "Manifolds/Stiefel/Stiefel.h"
#include "Manifolds/Euclidean/Euclidean.h"
#include "DPGO_types.h"
#include <Eigen/Dense>

/*Define the namespace*/
namespace DPGO{

  class LiftedSEVariable{
  public:
    LiftedSEVariable(int r, int d, int n);

    ~LiftedSEVariable();

    ROPTLIB::ProductElement* var(){
    	return MyVariable;
    }

    /**
    Write data to matrix
    */
    Matrix getData();

    /** 
	  Set data from matrix
    */
    void setData(const Matrix& Y);

  private:
  	ROPTLIB::StieVariable* StiefelVariable;
  	ROPTLIB::EucVariable* EuclideanVariable;
  	ROPTLIB::ProductElement* CartanVariable;
  	ROPTLIB::ProductElement* MyVariable;

  };
} /*end of SESync namespace*/


#endif