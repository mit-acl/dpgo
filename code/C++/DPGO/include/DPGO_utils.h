#ifndef DPGOUTILS_H
#define DPGOUTILS_H


#include "DPGO_types.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace DPGO{

Matrix projectToRotationGroup(const Matrix& M);


}





#endif