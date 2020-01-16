#include "DPGO_utils.h"


namespace DPGO{

	/**
	###############################################################
	############################################################### 
	The following implementations are originally from Cartan-Sync:
	https://bitbucket.org/jesusbriales/cartan-sync/src
	############################################################### 
	############################################################### 
	*/
	void constructOrientedConnectionIncidenceMatrixSE(const std::vector<RelativeSEMeasurement>& measurements, SparseMatrix& AT, DiagonalMatrix& OmegaT )
	{
		// Deduce graph dimensions from measurements
		size_t d; // Dimension of Euclidean space
		d = (!measurements.empty() ? measurements[0].t.size() : 0);
		size_t dh = d+1; // Homogenized dimension of Euclidean space
		size_t m; // Number of measurements
		m = measurements.size();
		size_t n = 0; // Number of poses
		for (const RelativeSEMeasurement &meas : measurements)
		{
			if (n < meas.p1) n = meas.p1;
			if (n < meas.p2) n = meas.p2;
		}
		n++; // Account for 0-based indexing: node indexes go from 0 to max({i,j})

		// Define connection incidence matrix dimensions
		// This is a [n x m] (dh x dh)-block matrix
		size_t rows = (d+1)*n;
		size_t cols = (d+1)*m;

		// We use faster ordered insertion, as suggested in
		// https://eigen.tuxfamily.org/dox/group__TutorialSparse.html#TutorialSparseFilling
		// TODO: Fix ColMajor (ours) or RowMajor (Rosen's)
		Eigen::SparseMatrix<double, Eigen::ColMajor> A(rows,cols); // default is column major
		// TODO: Actually for SE(d) matrices dimensions are 2x (3,3,3,4)
		// TODO: For our current formulation the 2nd matrix is Id (1nnz / col)
		A.reserve(Eigen::VectorXi::Constant(cols,8));
		DiagonalMatrix Omega(cols); // One block per measurement: (d+1)*m
		DiagonalMatrix::DiagonalVectorType &diagonal = Omega.diagonal();

		// Insert actual measurement values
		size_t i, j;
		for (size_t k = 0; k < m; k++) {
			const RelativeSEMeasurement &meas = measurements[k];
			i = meas.p1;
			j = meas.p2;

			/// Assign SE(d) matrix to block leaving node i
			/// AT(i,k) = -Tij (NOTE: NEGATIVE)
			// Do it column-wise for speed
			// Elements of rotation
			for (size_t c = 0; c < d; c++)
				for (size_t r = 0; r < d; r++)
			    	A.insert(i*dh+r,k*dh+c) = -meas.R(r,c);
			
			// Elements of translation
			for (size_t r = 0; r < d; r++)
				A.insert(i*dh+r,k*dh+d) = -meas.t(r);
			
			// Additional 1 for homogeneization
			A.insert(i*dh+d,k*dh+d) = -1;

			/// Assign (d+1)-identity matrix to block leaving node j
			/// AT(j,k) = +I (NOTE: POSITIVE)
			for (size_t r = 0; r < d+1; r++)
				A.insert(j*dh+r,k*dh+r)   = +1;

			/// Assign isotropic weights in diagonal matrix
			for (size_t r = 0; r < d; r++)
			  diagonal[k*dh+r] = meas.kappa;
			
			diagonal[k*dh+d]   = meas.tau;
		}

		A.makeCompressed();

		AT = A;
		OmegaT = Omega;
	}


	SparseMatrix constructConnectionLaplacianSE(const std::vector<RelativeSEMeasurement>& measurements){
		SparseMatrix AT;
		DiagonalMatrix OmegaT;
		constructOrientedConnectionIncidenceMatrixSE(measurements, AT, OmegaT);
		return AT * OmegaT * AT.transpose();
	}



	/**
	############################################################### 
	###############################################################  
	The following implementations are originally from SE-Sync: 
	https://github.com/david-m-rosen/SE-Sync.git
	############################################################### 
	############################################################### 
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

