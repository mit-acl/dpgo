/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef DPGOROBUST_H
#define DPGOROBUST_H


/**
Robust M-estimators used by DPGO. 
Notation (see Table 1 below): 
Parameter Estimation Techniques: A Tutorial with Application to Conic Fitting
https://www.microsoft.com/en-us/research/wp-content/uploads/2016/11/RR-2676.pdf
*/

namespace DPGO{

	
	class MEstimator{
		public:
			MEstimator(){}

			virtual ~MEstimator(){}
			
			/**
			Cost function given the residual x
			*/
			virtual double cost(double x) const = 0;

			/**
			Influence function (first derivative of cost)
			*/
			virtual double influence(double x) const = 0;

			/**
			Weight function (influence over x)
			*/
			virtual double weight(double x) const = 0;
	};

	

	class MEstimatorCauchy: public MEstimator{
		public:
			MEstimatorCauchy():c(1.0){}
			MEstimatorCauchy(double cIn):c(cIn){}

			virtual double cost(double x) const;
			virtual double influence(double x) const;
			virtual double weight(double x) const;

		private:
			double c;
	};



	class MEstimatorL2: public MEstimator{
		public:
			virtual double cost(double x) const;
			virtual double influence(double x) const;
			virtual double weight(double x) const;

	};


	class MEstimatorTruncatedL2: public MEstimator{
		public:
			MEstimatorTruncatedL2():c(2.0){}
			MEstimatorTruncatedL2(double cIn):c(cIn){}

			virtual double cost(double x) const;
			virtual double influence(double x) const;
			virtual double weight(double x) const;

		private:
			double c;
	};

}


#endif