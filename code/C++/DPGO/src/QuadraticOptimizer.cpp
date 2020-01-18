#include "QuadraticOptimizer.h"
#include <iostream>
#include "RTRNewton.h"
#include "RSD.h"
#include "SolversLS.h"

namespace DPGO{

	QuadraticOptimizer::QuadraticOptimizer(QuadraticProblem* p):problem(p){}

	QuadraticOptimizer::~QuadraticOptimizer(){}

	Matrix QuadraticOptimizer::optimize(const Matrix& Y) const{
		unsigned r = problem->relaxation_rank();
		unsigned d = problem->dimension();
		unsigned n = problem->num_poses();
		
		LiftedSEVariable VarInit(r,d,n);
		VarInit.setData(Y);
		VarInit.var()->NewMemoryOnWrite();


		// Use RTR 
		// ROPTLIB::RTRNewton Solver(problem, VarInit.var());
		// Solver.Stop_Criterion = ROPTLIB::StopCrit::FUN_REL;
		// Solver.maximum_Delta = 1e4;
		// Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
		// Solver.Max_Iteration = 1;
		// Solver.Run();

		// Use RGD
		ROPTLIB::RSD Solver(problem, VarInit.var());
		Solver.Max_Iteration = 1;
		Solver.Debug = ROPTLIB::DEBUGINFO::ITERRESULT;
		Solver.Run();

		// Retrieve results
		const ROPTLIB::ProductElement *Yopt = static_cast<const ROPTLIB::ProductElement*>(Solver.GetXopt());
		LiftedSEVariable VarOpt(r,d,n);
		Yopt->CopyTo(VarOpt.var());

		cout << "Initial objective value: " << problem->f(VarInit.var()) << endl;
		cout << "Final objective value: " << Solver.Getfinalfun() << endl;
		cout << "Final gradient norm: " << Solver.Getnormgf() << endl;

		return VarOpt.getData();
	}



}