#ifndef DPGO_TYPES_H
#define DPGO_TYPES_H

#include <tuple>
#include <map>
#include <Eigen/Core>
#include <Eigen/SparseCore>

namespace DPGO {

typedef Eigen::VectorXd Vector;
typedef Eigen::MatrixXd Matrix;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagonalMatrix;
typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SparseMatrix;

/** 
	Riemannian optimization algorithms from ROPTLIB
	to be used as solver.
*/
enum ROPTALG{
	// Riemannian Trust-Region (RTRNewton in ROPTLIB)
	RTR, 

	// Riemannian gradient descent (RSD in ROPTLIB)
	RGD
};


// In distributed PGO, each pose is uniquely determined by the robot ID and pose ID
typedef std::pair<unsigned, unsigned> PoseID;

// Implement a dictionary for easy access of pose value by PoseID
typedef std::map<PoseID, Matrix, std::less<PoseID>, 
      Eigen::aligned_allocator<std::pair<PoseID, Matrix>>> PoseDict;

} 

#endif 
