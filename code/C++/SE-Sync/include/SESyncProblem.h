/** This class encapsulates an instance of the special Euclidean synchronization
* problem
*
* dmrosen 18 May 2017
* jbriales 24 Aug 2017
*/

#ifndef _SESYNCPROBLEM_H_
#define _SESYNCPROBLEM_H_

#include "Problem.h"

namespace SESync {

class SESyncProblem : public ROPTLIB::Problem {
// private: // TODO: can be set only from methods in this parent class
protected:
  /** Number of poses */
  unsigned int n = 0;

  /** Number of measurements */
  unsigned int m = 0;

  /** Dimensionality of the Euclidean space */
  unsigned int d = 0;

  /** The rank of the rank-restricted relaxation */
  unsigned int r = 0;

public:
  /** Set the maximum rank of the rank-restricted semidefinite relaxation
      and set the corresponding optimization domain */
  virtual void set_relaxation_rank(unsigned int rank) =0;

  /// ACCESSORS

  unsigned int num_poses() const { return n; }

  unsigned int num_measurements() const { return m; }

  unsigned int dimension() const { return d; }

  /** Get the rank of the rank-restricted semidefinite relaxation */
  unsigned int relaxation_rank() const { return r; }

  /// OVERRIDDEN PURE VIRTUAL BASE CLASS (ROPTLIB::PROBLEM) FUNCTIONS

  /** Evaluates the problem objective */
  double f(ROPTLIB::Variable* x) const =0;

  /** Evaluates the Euclidean gradient of the function */
  void EucGrad(ROPTLIB::Variable* x, ROPTLIB::Vector* g) const =0;

  /** Evaluates the action of the Euclidean Hessian of the function */
  void EucHessianEta(ROPTLIB::Variable* x, ROPTLIB::Vector* v,
                     ROPTLIB::Vector* Hv) const =0;
};

}
#endif  // _SESYNCPROBLEM_H_
