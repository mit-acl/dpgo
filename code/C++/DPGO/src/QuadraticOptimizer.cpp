#include "QuadraticOptimizer.h"
#include <iostream>
#include <stdexcept>
#include "RTRNewton.h"
#include "RSD.h"
#include "SolversLS.h"

namespace DPGO{

	QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem* p):
	problem(p), algorithm(ROPTALG::RTR), stepsize(1e-3), verbose(true){}

	QuadraticOptimizer::~QuadraticOptimizer(){}

	Matrix QuadraticOptimizer::optimize(const Matrix& Y){
		if(algorithm == ROPTALG::RTR)
		{	
			return trustRegion(Y);
		}
		else
		{	
			assert(algorithm == ROPTALG::RGD);
			return gradientDescent(Y);
		}	
	}

	Matrix QuadraticOptimizer::trustRegion(const Matrix& Yinit){
		unsigned r = problem->relaxation_rank();
		unsigned d = problem->dimension();
		unsigned n = problem->num_poses();

		LiftedSEVariable VarInit(r,d,n);
		VarInit.setData(Yinit);
		VarInit.var()->NewMemoryOnWrite();

		ROPTLIB::RTRNewton Solver(problem, VarInit.var());
		Solver.Stop_Criterion = ROPTLIB::StopCrit::GRAD_F; // Stoping criterion based on gradient norm
		Solver.Tolerance = 1e-2; // Tolerance associated with stopping criterion
		Solver.maximum_Delta = 1e3; // Maximum trust-region radius
		Solver.Debug = (verbose ? ROPTLIB::DEBUGINFO::DETAILED : ROPTLIB::DEBUGINFO::NOOUTPUT);
		Solver.Max_Iteration = 10; // Max RTR iterations
		Solver.Max_Inner_Iter = 100; // Max tCG iterations
		Solver.Run();
		
		const ROPTLIB::ProductElement *Yopt = static_cast<const ROPTLIB::ProductElement*>(Solver.GetXopt());
		LiftedSEVariable VarOpt(r,d,n);
		Yopt->CopyTo(VarOpt.var());
		if (verbose){
			cout << "Initial objective value: " << problem->f(VarInit.var()) << endl;
			cout << "Final objective value: " << Solver.Getfinalfun() << endl;
			cout << "Final gradient norm: " << Solver.Getnormgf() << endl;
		}

		return VarOpt.getData();
	}


	Matrix QuadraticOptimizer::gradientDescent(const Matrix& Yinit)
	{
		unsigned r = problem->relaxation_rank();
		unsigned d = problem->dimension();
		unsigned n = problem->num_poses();

		LiftedSEManifold M(r,d,n);
		LiftedSEVariable VarInit(r,d,n);
		LiftedSEVariable VarNext(r,d,n);
		LiftedSEVector RGrad(r,d,n);
		VarInit.setData(Yinit);

		// Euclidean gradient
		problem->EucGrad(VarInit.var(), RGrad.vec());

		// Riemannian gradient
		M.getManifold()->Projection(VarInit.var(), RGrad.vec(), RGrad.vec());

		// Preconditioning
		// problem->PreConditioner(VarInit.var(), RGrad.vec(), RGrad.vec());

		// Update
		M.getManifold()->ScaleTimesVector(VarInit.var(), -stepsize, RGrad.vec(), RGrad.vec());
		M.getManifold()->Retraction(VarInit.var(), RGrad.vec(), VarNext.var());

		return VarNext.getData();
	}

	Matrix QuadraticOptimizer::gradientDescentLS(const Matrix& Yinit){
		unsigned r = problem->relaxation_rank();
		unsigned d = problem->dimension();
		unsigned n = problem->num_poses();

		LiftedSEVariable VarInit(r,d,n);
		VarInit.setData(Yinit);

		ROPTLIB::RSD Solver(problem, VarInit.var());
		Solver.Stop_Criterion = ROPTLIB::StopCrit::GRAD_F; 
		Solver.Tolerance = 1e-2; 
		Solver.Max_Iteration = 10; 
		Solver.Debug = (verbose ? ROPTLIB::DEBUGINFO::DETAILED : ROPTLIB::DEBUGINFO::NOOUTPUT);
		Solver.Run();
		
		const ROPTLIB::ProductElement *Yopt = static_cast<const ROPTLIB::ProductElement*>(Solver.GetXopt());
		LiftedSEVariable VarOpt(r,d,n);
		Yopt->CopyTo(VarOpt.var());
		if (verbose){
			cout << "Initial objective value: " << problem->f(VarInit.var()) << endl;
			cout << "Final objective value: " << Solver.Getfinalfun() << endl;
			cout << "Final gradient norm: " << Solver.Getnormgf() << endl;
		}

		return VarOpt.getData();
	}

}