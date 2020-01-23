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
		Matrix optimize(const Matrix& Y);

		/**
		Optimize using chordal initialization
		*/
		Matrix optimize();

		/**
		Turn on/off verbose output
		*/
		void setVerbose(bool v) {verbose = v;}


		/**
		Set optimization algorithm
		*/
		void setAlgorithm(ROPTALG alg) {algorithm = alg;}


		/**
		Set maximum step size
		*/
		void setStepsize(double s) {stepsize = s;}

	private:
		// Underlying Riemannian Optimization Problem
		QuadraticProblem* problem;

		// Optimization algorithm to be used
		ROPTALG algorithm;

		// step size (only for RGD)
		double stepsize;

		// Verbose flag
		bool verbose;

		// Custom implementation of constant step size RGD
		Matrix optimizeRGD(const Matrix& Yinit);
	};

}




#endif