#include <iostream>
#include "QuadraticProblem.h"


using namespace std;

/*Define the namespace*/
namespace DPGO{

	QuadraticProblem::QuadraticProblem(unsigned int nIn, unsigned int dIn, unsigned int rIn, SparseMatrix& QIn, SparseMatrix& GIn):
	Q(QIn), G(GIn), n(nIn), d(dIn), r(rIn)
	{
		
		SetUseGrad(true);
    	SetUseHess(true);

    	M = new LiftedSEManifold(r,d,n);
		Variable = new LiftedSEVariable(r,d,n);
		Vector = new LiftedSEVector(r,d,n);
		HessianVectorProduct = new LiftedSEVector(r,d,n);

		SetDomain(M->getManifold());

		SparseMatrix P = Q;	
		for(unsigned row = 0; row < P.rows(); ++row){
			P.coeffRef(row,row) += 0.01;
		}
		solver.compute(P);
		if(solver.info() != Eigen::Success){
			cout << "WARNING: Precon.compute() failed." << endl;
		}
	}

	QuadraticProblem::~QuadraticProblem()
	{
		delete Variable;
		delete Vector;
		delete HessianVectorProduct;
		delete M;
	}

	double QuadraticProblem::f(ROPTLIB::Variable* x) const 
	{
		x->CopyTo(Variable->var());
		Matrix Y = Variable->getData();
		return 0.5 * (Y * Q * Y.transpose()).trace() + (Y * G.transpose()).trace();
	}

	void QuadraticProblem::EucGrad(ROPTLIB::Variable* x, ROPTLIB::Vector* g) const 
	{
		x->CopyTo(Variable->var());
		Matrix Y = Variable->getData();
		Matrix EGrad = Y * Q + G;
		Vector->setData(EGrad);
		Vector->vec()->CopyTo(g);
	}

	void QuadraticProblem::EucHessianEta(ROPTLIB::Variable* x, ROPTLIB::Vector* v,
                 ROPTLIB::Vector* Hv) const
	{
		v->CopyTo(Vector->vec());
		Matrix inVec = Vector->getData();
		Matrix outVec = inVec * Q;
		HessianVectorProduct->setData(outVec);
		HessianVectorProduct->vec()->CopyTo(Hv);
	}

	void QuadraticProblem::PreConditioner(ROPTLIB::Variable* x, ROPTLIB::Vector* inVec,
                      ROPTLIB::Vector* outVec) const 
	{

		inVec->CopyTo(Vector->vec());
		Matrix HV = solver.solve(Vector->getData().transpose()).transpose();
		if(solver.info() != Eigen::Success){
			cout << "WARNING: Precon.solve() failed." << endl;
		}
		// Project to tangent space
		x->CopyTo(Variable->var());
		Vector->setData(HV);
		M->getManifold()->Projection(Variable->var(), Vector->vec(), Vector->vec());
		Vector->vec()->CopyTo(outVec);

	}
}