#include "manifold/LiftedSEVector.h"


using namespace std;
using namespace ROPTLIB;

namespace DPGO{

	LiftedSEVector::LiftedSEVector(int r, int d, int n){
		StiefelVector = new StieVector(r,d);
		EuclideanVector = new EucVector(r);
		CartanVector = new ProductElement(2, StiefelVector, 1, EuclideanVector, 1);
		MyVector = new ProductElement(1, CartanVector, n);
	}

	LiftedSEVector::~LiftedSEVector(){
		// Avoid memory leak
		delete StiefelVector;
		delete EuclideanVector;
		delete CartanVector;
		delete MyVector;
	}

	Matrix LiftedSEVector::getData(){
		ProductElement* T = static_cast<ProductElement*>(MyVector->GetElement(0));
	  	const int *sizes = T->GetElement(0)->Getsize();
	  	unsigned int r = sizes[0];
	  	unsigned int d = sizes[1];
	  	unsigned int n = MyVector->GetNumofElement();
	  	return Eigen::Map<Matrix>((double *)MyVector->ObtainReadData(), r, n*(d+1));
		}

    void LiftedSEVector::setData(const Matrix& Y){
    	ProductElement* T = static_cast<ROPTLIB::ProductElement*>(MyVector->GetElement(0));
	    const int *sizes = T->GetElement(0)->Getsize();
	    unsigned int r = sizes[0];
	    unsigned int d = sizes[1];
	    unsigned int n = MyVector->GetNumofElement();

	    // Copy array data from Eigen matrix to ROPTLIB variable
	    const double *matrix_data = Y.data();
	    double *prodvar_data = MyVector->ObtainWriteEntireData();
	    memcpy(prodvar_data, matrix_data, sizeof(double) * r * (d+1) * n);
    }

}