#ifndef DPGOUTILS_H
#define DPGOUTILS_H


#include "DPGO_types.h"
#include "RelativeSEMeasurement.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace DPGO{

	void constructOrientedConnectionIncidenceMatrixSE(const std::vector<RelativeSEMeasurement>& measurements, SparseMatrix& AT, DiagonalMatrix& OmegaT);

	SparseMatrix constructConnectionLaplacianSE(const std::vector<RelativeSEMeasurement>& measurements);

	Matrix projectToRotationGroup(const Matrix& M);

	
}





#endif