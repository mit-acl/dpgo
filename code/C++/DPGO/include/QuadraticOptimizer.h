/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

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

		// Apply RTR
		Matrix trustRegion(const Matrix& Yinit);

		// Apply a single RGD iteration with constant step size
		Matrix gradientDescent(const Matrix& Yinit);

		// Apply gradient descent with line search
		Matrix gradientDescentLS(const Matrix& Yinit);

	};

}




#endif