/* ----------------------------------------------------------------------------
 * Copyright 2020, Massachusetts Institute of Technology, * Cambridge, MA 02139
 * All Rights Reserved
 * Authors: Yulun Tian, et al. (see README for the full author list)
 * See LICENSE for the license information
 * -------------------------------------------------------------------------- */

#include <DPGO/multithread/RGDWorker.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <random>

using namespace std;

namespace DPGO {

RGDWorker::RGDWorker(RGDMaster* pMaster, unsigned pId)
    : master(pMaster),
      id(pId),
      mFinishRequested(false),
      mFinished(false),
      rate(200.0),  // default rate is 200 Hz
      stepsize(0.001) {
  d = master->dimension();
  r = master->relaxation_rank();

  M = new LiftedSEManifold(r, d, 1);
  Var = new LiftedSEVariable(r, d, 1);
  VarNext = new LiftedSEVariable(r, d, 1);
  EGrad = new LiftedSEVector(r, d, 1);
  RGrad = new LiftedSEVector(r, d, 1);
  Eta = new LiftedSEVector(r, d, 1);

  cout << "Worker " << id << " initialized. " << endl;
}

RGDWorker::~RGDWorker() {
  delete M;
  delete Var;
  delete VarNext;
  delete EGrad;
  delete RGrad;
  delete Eta;
}

void RGDWorker::run() {
  std::random_device
      rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 rng(rd());  // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> UniformDistribution(0,
                                                      updateIndices.size() - 1);
  std::exponential_distribution<double> ExponentialDistribution(rate);

  auto startTime = std::chrono::high_resolution_clock::now();
  double numWrites = 0.0;

  while (true) {
    // randomly select an index
    unsigned i = updateIndices[UniformDistribution(rng)];

    Matrix Yi = readComponent(i);

    Matrix Gi = computeEuclideanGradient(i);

    Matrix YiNext = gradientUpdate(Yi, Gi);

    writeComponent(i, YiNext);

    numWrites += 1.0;

    if (mFinishRequested) break;

    // use usleep for microsecond
    double sleepUs = 1e6 * ExponentialDistribution(rng);
    usleep(sleepUs);
  }

  auto counter = std::chrono::high_resolution_clock::now() - startTime;
  double elapsedMs =
      std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();

  mFinished = true;
  cout << "Worker " << id
       << " finished. Average runtime per iteration (sleep included): "
       << elapsedMs / numWrites << " milliseconds." << endl;
}

void RGDWorker::requestFinish() { mFinishRequested = true; }

bool RGDWorker::isFinished() { return mFinished; }

Matrix RGDWorker::readDataMatrixBlock(unsigned i, unsigned j) {
  return master->readDataMatrixBlock(i, j);
}

Matrix RGDWorker::readComponent(unsigned i) {
  // obtain lock
  lock_guard<mutex> lock(master->mUpdateMutexes[i]);
  return master->readComponent(i);
}

void RGDWorker::writeComponent(unsigned i, const Matrix& Yi) {
  // obtain lock
  lock_guard<mutex> lock(master->mUpdateMutexes[i]);
  master->writeComponent(i, Yi);
}

Matrix RGDWorker::computeEuclideanGradient(unsigned i) {
  Matrix Gi(r, d + 1);
  Gi.setZero();
  // iterate over neighbors of i
  for (unsigned k = 0; k < master->adjList[i].size(); ++k) {
    unsigned j = master->adjList[i][k];
    Matrix Yj, Qji;
    Yj = readComponent(j);
    Qji = readDataMatrixBlock(j, i);
    Gi = Gi + Yj * Qji;
  }
  return Gi;
}

Matrix RGDWorker::gradientUpdate(const Matrix& Yi, const Matrix& Gi) {
  Var->setData(Yi);
  EGrad->setData(Gi);
  M->getManifold()->Projection(Var->var(), EGrad->vec(), RGrad->vec());
  M->getManifold()->ScaleTimesVector(Var->var(), -stepsize, RGrad->vec(),
                                     Eta->vec());
  M->getManifold()->Retraction(Var->var(), Eta->vec(), VarNext->var());
  return VarNext->getData();
}
}  // namespace DPGO