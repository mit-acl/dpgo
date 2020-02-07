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


		QuadraticProblem(const unsigned int nIn, const unsigned int dIn, const unsigned int rIn, const SparseMatrix& QIn, const SparseMatrix& GIn);


		virtual ~QuadraticProblem();


		/** Number of pose variables */
		unsigned int num_poses() const { return n; }


		/** Dimension (2 or 3) of estimation problem */ 
		unsigned int dimension() const { return d; }


		/** Relaxation rank in Riemannian optimization problem */ 
		unsigned int relaxation_rank() const { return r; }


		/** Evaluates the problem objective */
		double f(const Matrix& Y) const ;


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

  		
  		/** Evaluate the norm of the Riemannian gradient for the given solution */
  		double gradNorm(const Matrix& Y) const;


  		/** The quadratic component of the cost function */
  		const SparseMatrix Q;


  		/** The linear component of the cost function */
		const SparseMatrix G;
  		
	private:

		/** Number of poses */
		unsigned int n = 0;

		/** Dimensionality of the Euclidean space */
		unsigned int d = 0;

		/** The rank of the rank-restricted relaxation */
  		unsigned int r = 0;

  		/** 
		Manifold object
		*/
  		LiftedSEManifold* M;

  		/** 
		Manifold variable
		*/
		LiftedSEVariable* Variable;
		
		/** 
		Tangent vectors
		*/
		LiftedSEVector* Vector;


		LiftedSEVector* HessianVectorProduct;

		/** 
		Solver used by preconditioner
		*/
		Eigen::CholmodDecomposition<SparseMatrix> solver;
		

	};

}


#endif