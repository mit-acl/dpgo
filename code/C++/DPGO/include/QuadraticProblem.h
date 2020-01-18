#ifndef QUADRATICPROBLEM_H
#define QUADRATICPROBLEM_H

#include <vector>
#include "Problem.h"
#include "manifold/LiftedSEManifold.h"
#include "manifold/LiftedSEVariable.h"
#include "manifold/LiftedSEVector.h"
#include "DPGO_types.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/CholmodSupport>

using namespace std;

/*Define the namespace*/
namespace DPGO{


	/** This class implements a ROPTLIB problem with the following cost function:
	    f(X) = 0.5*<Q, XtX> + <X,G>
	    Q is the quadratic part with dimension (d+1)n-by-(d+1)n
	    G is the linear part with dimension r-by-(d+1)n
	*/
	class QuadraticProblem : public ROPTLIB::Problem{

	public:
		/** Default constructor; doesn't actually do anything */
  		QuadraticProblem(){}

		QuadraticProblem(unsigned int nIn, unsigned int dIn, unsigned int rIn, SparseMatrix& QIn, SparseMatrix& GIn);

		virtual ~QuadraticProblem();

		unsigned int num_poses() const { return n; }

		unsigned int dimension() const { return d; }

		unsigned int relaxation_rank() const { return r; }

  		/** Evaluates the problem objective */
		double f(ROPTLIB::Variable* x) const ;

		/** Evaluates the Euclidean gradient of the function */
		void EucGrad(ROPTLIB::Variable* x, ROPTLIB::Vector* g) const ;

		/** Evaluates the action of the Euclidean Hessian of the function */
		void EucHessianEta(ROPTLIB::Variable* x, ROPTLIB::Vector* v,
                     ROPTLIB::Vector* Hv) const ;

		/** Evaluates the action of the Preconditioner for the Hessian of the function */
  		void PreConditioner(ROPTLIB::Variable* x, ROPTLIB::Vector* inVec,
                      ROPTLIB::Vector* outVec) const ;

  		
  		SparseMatrix Q;
		SparseMatrix G;
  		
	private:
		/** Number of poses */
		unsigned int n = 0;

		/** Dimensionality of the Euclidean space */
		unsigned int d = 0;

		/** The rank of the rank-restricted relaxation */
  		unsigned int r = 0;

  		LiftedSEManifold* M;

		LiftedSEVariable* Variable;
		
		LiftedSEVector* Vector;

		LiftedSEVector* HessianVectorProduct;

		// Solver used for preconditioner
		Eigen::CholmodDecomposition<SparseMatrix> solver;
		

	};

}


#endif