#ifndef QUADRATICOPTIMIZER_H
#define QUADRATICOPTIMIZER_H

#include "manifold/LiftedSEManifold.h"
#include "manifold/LiftedSEVariable.h"
#include "manifold/LiftedSEVector.h"
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"

namespace DPGO{

	class QuadraticOptimizer{
	public:
		QuadraticOptimizer(QuadraticProblem* p);
		
		~QuadraticOptimizer();

		/**
		Optimize from the given initial guess
		*/
		Matrix optimize(const Matrix& Y) const;

		/**
		Optimize using chordal initialization
		*/
		Matrix optimize() const;

	private:
		// Underlying Riemannian Optimization Problem
		QuadraticProblem* problem;

	};

}




#endif