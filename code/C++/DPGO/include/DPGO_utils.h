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
	Given a vector of relative pose measurements, this function computes and returns the B matrices defined in equation (69) of the tech report 
	*/
	void constructBMatrices(const std::vector<RelativeSEMeasurement>& measurements, SparseMatrix& B1, SparseMatrix& B2, SparseMatrix& B3);


	/** 
	Given the measurement matrix B3 defined in equation (69c) of the tech report and the problem dimension d, this function computes and returns the corresponding chordal initialization for the rotational states 
	*/
	Matrix chordalInitialization(unsigned int d, const SparseMatrix& B3);


	/** 
	Given the measurement matrices B1 and B2 and a matrix R of rotational state estimates, this function computes and returns the corresponding optimal translation estimates 
	*/
	Matrix recoverTranslations(const SparseMatrix& B1, const SparseMatrix& B2, const Matrix& R);


	/**
	Project a given matrix to the rotation group
	*/
	Matrix projectToRotationGroup(const Matrix& M);

}





#endif