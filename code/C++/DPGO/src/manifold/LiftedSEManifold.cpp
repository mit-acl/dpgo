#include "manifold/LiftedSEManifold.h"

using namespace std;
using namespace ROPTLIB;

namespace DPGO{
	LiftedSEManifold::LiftedSEManifold(int r, int d, int n){
		StiefelManifold = new Stiefel(r,d);
		StiefelManifold->ChooseStieParamsSet3();
		EuclideanManifold = new Euclidean(r);
		CartanManifold = new ProductManifold(2, StiefelManifold, 1, EuclideanManifold, 1);
		MyManifold = new ProductManifold(1, CartanManifold, n);
	}

	LiftedSEManifold::~LiftedSEManifold(){
		// Avoid memory leak
		delete StiefelManifold;
		delete EuclideanManifold;
		delete CartanManifold;
		delete MyManifold;
	}

}