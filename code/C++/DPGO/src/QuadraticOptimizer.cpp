#include "QuadraticOptimizer.h"
#include <iostream>
#include "RTRNewton.h"
#include "RSD.h"
#include "SolversLS.h"

namespace DPGO{

	QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem* p):
	problem(p), algorithm(ROPTALG::RTR), maxStepsize(1e2), verbose(true){}

	QuadraticOptimizer::~QuadraticOptimizer(){}

	Matrix QuadraticOptimizer::optimize(const Matrix& Y) const{
		unsigned r = problem->relaxation_rank();
		unsigned d = problem->dimension();
		unsigned n = problem->num_poses();
		
		LiftedSEVariable VarInit(r,d,n);
		VarInit.setData(Y);
		VarInit.var()->NewMemoryOnWrite();

		if(algorithm == ROPTALG::RTR)
		{	
			ROPTLIB::RTRNewton Solver(problem, VarInit.var());
			Solver.Stop_Criterion = ROPTLIB::StopCrit::FUN_REL;
			Solver.maximum_Delta = 1e1;
			Solver.Debug = (verbose ? ROPTLIB::DEBUGINFO::ITERRESULT : ROPTLIB::DEBUGINFO::NOOUTPUT);
			Solver.Max_Iteration = 10;
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
		else
		{
			ROPTLIB::RSD Solver(problem, VarInit.var());
			Solver.Max_Iteration = 1;
			Solver.Maxstepsize = maxStepsize;
			Solver.Debug = (verbose ? ROPTLIB::DEBUGINFO::ITERRESULT : ROPTLIB::DEBUGINFO::NOOUTPUT);		
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
}