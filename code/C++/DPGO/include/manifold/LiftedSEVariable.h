#ifndef LIFTEDSEVARIABLE_H
#define LIFTEDSEVARIABLE_H

#include "Manifolds/ProductElement.h"
#include "Stiefel.h"
#include "Euclidean.h"
#include "DPGO_types.h"
#include <Eigen/Dense>

/*Define the namespace*/
namespace DPGO{

  class LiftedSEVariable{
  public:
    LiftedSEVariable(int r, int d, int n);

    ~LiftedSEVariable();


    /**
    Write data to matrix
    */
    void getData(Matrix& Y);

    /** 
	Set data from matrix
    */
    void setData(Matrix& Y);

  private:
  	ROPTLIB::StieVariable* StiefelVariable;
  	ROPTLIB::EucVariable* EuclideanVariable;
  	ROPTLIB::ProductElement* CartanVariable;
  	ROPTLIB::ProductElement* MyVariable;

  };
} /*end of SESync namespace*/


#endif