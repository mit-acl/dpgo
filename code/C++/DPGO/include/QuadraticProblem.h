#ifndef QUADRATICPROBLEM_H
#define QUADRATICPROBLEM_H

#include <vector>
#include <Eigen/Dense>
#include "SESync.h"
#include "SESync_utils.h"
#include "SESync_types.h"

using namespace std;
using namespace SESync;

/*Define the namespace*/
namespace DPGO{

	class QuadraticProblem{

	public:
		QuadraticProblem(unsigned int nIn, unsigned int dIn, unsigned int rIn, SparseMatrix& QIn, SparseMatrix& GIn);

		unsigned int num_poses() const { return n; }

  		unsigned int dimension() const { return d; }

  		unsigned int relaxation_rank() const { return r; }

  		SparseMatrix Q;
		SparseMatrix G;
  		
	private:
		/** Number of poses */
		unsigned int n = 0;

		/** Dimensionality of the Euclidean space */
		unsigned int d = 0;

		/** The rank of the rank-restricted relaxation */
  		unsigned int r = 0;

		/** Data matrices that define the cost function 
		    f(X) = 0.5*<Q, XtX> + <X,G>
		    Q is the quadratic part with dimension (d+1)n-by-(d+1)n
		    G is the linear part with dimension r-by-(d+1)n
		*/
		

	};

}


#endif