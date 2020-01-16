#include "DPGO_utils.h"


namespace DPGO{


	/** This implementation is originally from SE-Sync: 
	https://github.com/david-m-rosen/SE-Sync.git
	*/
	Matrix projectToRotationGroup(const Matrix &M) {
		// Compute the SVD of M
		Eigen::JacobiSVD<Matrix> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

		double detU = svd.matrixU().determinant();
		double detV = svd.matrixV().determinant();

		if (detU * detV > 0) {
			return svd.matrixU() * svd.matrixV().transpose();
		} else {
			Eigen::MatrixXd Uprime = svd.matrixU();
			Uprime.col(Uprime.cols() - 1) *= -1;
			return Uprime * svd.matrixV().transpose();
		}
	}



}

