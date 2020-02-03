#ifndef DPGOUTILS_H
#define DPGOUTILS_H


#include "DPGO_types.h"
#include "RelativeSEMeasurement.h"
#include <Eigen/Dense>
#include <Eigen/SVD>

namespace DPGO{

	/**
	Helper function to read a dataset in .g2o format
	*/
	std::vector<RelativeSEMeasurement> read_g2o_file(const std::string& filename, size_t& num_poses);

	/**
	Helper function to construct connection laplacian matrix in SE(d)
	*/
	void constructOrientedConnectionIncidenceMatrixSE(const std::vector<RelativeSEMeasurement>& measurements, SparseMatrix& AT, DiagonalMatrix& OmegaT);

	/**
	Helper function to construct connection laplacian matrix in SE(d) 
	*/
	SparseMatrix constructConnectionLaplacianSE(const std::vector<RelativeSEMeasurement>& measurements);


	/**
	Project a given matrix to the rotation group
	*/
	Matrix projectToRotationGroup(const Matrix& M);

}





#endif