/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#ifndef RGDMASTER_H
#define RGDMASTER_H

#include <DPGO/QuadraticProblem.h>
#include <DPGO/manifold/LiftedSEManifold.h>
#include <DPGO/manifold/LiftedSEVariable.h>
#include <DPGO/manifold/LiftedSEVector.h>
#include <DPGO/multithread/RGDWorker.h>

#include <Eigen/Dense>
#include <mutex>
#include <thread>
#include <vector>

#include "Manifolds/Element.h"
#include "Manifolds/Manifold.h"

using namespace std;

/*Define the namespace*/
namespace DPGO {

class RGDWorker;

class RGDMaster {
 public:
  RGDMaster(QuadraticProblem* p, Matrix Y0);

  ~RGDMaster();

  void solve(unsigned num_threads);

  void setUpdateRate(int freq) { updateRate = freq; }

  void setStepsize(float s) { stepsize = s; }

  Matrix readDataMatrixBlock(unsigned i, unsigned j);

  Matrix readComponent(unsigned i);

  void writeComponent(unsigned i, const Matrix& Yi);

  void getSolution(Matrix& Yout) { Yout = Y; }

  unsigned int num_poses() const { return n; }

  unsigned int dimension() const { return d; }

  unsigned int relaxation_rank() const { return r; }

  // mutexes for each coordinate (to ensure atomic read/write)
  vector<mutex> mUpdateMutexes;

  // adjacency list between coordinates
  vector<vector<unsigned>> adjList;

  // number of writes performed by ALL workers
  unsigned numWrites;

 private:
  // problem object
  QuadraticProblem* problem = nullptr;

  // current iterate
  Matrix Y;

  // step size
  float stepsize;

  // update rate in Hz
  int updateRate;

  // problem specific constants
  unsigned d, r, n;

  // list of workers
  vector<thread*> threads;
  vector<RGDWorker*> workers;

  // ROPTLIB
  LiftedSEManifold* M;
  LiftedSEVariable* Var;
  LiftedSEVector* EGrad;
  LiftedSEVector* RGrad;

  void initialize();

  float computeCost();

  float computeGradNorm();
};

}  // namespace DPGO

#endif