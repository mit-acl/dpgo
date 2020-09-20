/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <cmath>
#include <DPGO/DPGO_robust.h>

using namespace std;

namespace DPGO{

	/**
	L2 
	*/
	double MEstimatorL2::cost(double x) const{
		return x*x/2;
	}

	double MEstimatorL2::influence(double x) const{
		return x;
	}

	double MEstimatorL2::weight(double x) const{
		return 1.0;
	}


	/**
	Cauchy
	*/ 
	double MEstimatorCauchy::cost(double x) const{
		return (c*c/2)*log(1 + (x/c)*(x/c));
	}

	double MEstimatorCauchy::influence(double x) const{
		return x / (1 + abs(x)/c);
	}

	double MEstimatorCauchy::weight(double x) const{
		return 1 / (1 + abs(x)/c);
	}


	/**
	Truncated L2
	*/ 
	double MEstimatorTruncatedL2::cost(double x) const{
		if (x > c){
			return 0;
		}
		return x*x/2;
	}

	double MEstimatorTruncatedL2::influence(double x) const{
		if (x > c){
			return 0;
		}
		return x;
	}

	double MEstimatorTruncatedL2::weight(double x) const{
		if (x > c){
			return 0;
		}
		return 1;
	}



}