#include <iostream>
#include "SESync.h"
#include "SESync_utils.h"
#include "QuadraticProblem.h"
#include "multithread/RGDMaster.h"

using namespace std;
using namespace DPGO;

int main(int argc, char** argv)
{
    if (argc < 2) {
    	cout << "Parallel asynchronous RGD for pose-graph optimization. " << endl;
        cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
        exit(1);
    }

    size_t num_poses;
    vector<SESync::RelativePoseMeasurement> measurements = SESync::read_g2o_file(argv[1], num_poses);
    cout << "Loaded dataset from file " << argv[1] << endl;

    SparseMatrix ConLapT = construct_connection_Laplacian_T(measurements);
    unsigned int n,d,r;
    d = (!measurements.empty() ? measurements[0].t.size() : 0);
    n = ConLapT.rows()/(d+1);
    r = 5;

    // Input pose-graph optimization problem is not anchored (global symmetry)
    // Hence there is no linear term in the cost function
    SparseMatrix G(r,(d+1)*n);
    G.setZero();
    QuadraticProblem* problem = new QuadraticProblem(n,d,r,ConLapT,G);

    Matrix Y;

    /** call SESync solver */
    SESyncOpts opts;
    opts.verbose = true; // Print output to stdout
    opts.eig_comp_tol = 1e-6; // 1e-10
    opts.min_eig_num_tol = 1e-3; // this is the value used in Matlab version
    SESyncResult results = SESync::SESync(measurements, AlgType::CartanSync, opts);
    // Y = results.Yopt;

    // Random initialization
    // CartanSyncVariable Yinit(r,problem->dimension(),problem->num_poses());
    // Yinit.RandInManifold();
    // Y.resize(r, problem->dimension() * problem->num_poses());
    // CartanProd2Mat(Yinit, Y);

    // Chordal initialization
    SparseMatrix B1, B2, B3; 
    construct_B_matrices(measurements, B1, B2, B3);
    Matrix Rinit = chordal_initialization(problem->dimension(), B3);
    Matrix tinit = recover_translations(B1, B2, Rinit);
    Y.resize(r, n*(d+1));
    Y.setZero();
    for (size_t i=0; i<n; i++)
    {
        Y.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Y.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
    }


    /** Call asynchronous PGO solver */
    // unsigned numberOfThreads[8] = {1,2,3,4,5,6,7,8};
    // unsigned numberOfTrials = 5;
    // vector<float> averageWritesVector;

    // for(unsigned i = 0; i < 8; ++i){
    //     unsigned averageNumberOfWrites = 0;
    //     for(unsigned k = 1; k <= numberOfTrials; ++k){
    //         RGDMaster master(problem, Y);
    //         master.solve(numberOfThreads[i]);
    //         averageNumberOfWrites += master.numWrites;
    //     }
    //     averageWritesVector.push_back((float) averageNumberOfWrites / (float) numberOfTrials);
    // }

    // for(unsigned i = 0; i < averageWritesVector.size(); ++i){
    //     cout << averageWritesVector[i] << endl;
    // }
    
    RGDMaster master(problem, Y);
    master.solve(8);

    exit(0);
}
