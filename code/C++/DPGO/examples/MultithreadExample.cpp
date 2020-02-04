#include <iostream>
#include "QuadraticProblem.h"
#include "DPGO_types.h"
#include "DPGO_utils.h"
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
    vector<RelativeSEMeasurement> measurements = read_g2o_file(argv[1], num_poses);
    cout << "Loaded dataset from file " << argv[1] << endl;

    SparseMatrix ConLapT = constructConnectionLaplacianSE(measurements);
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

    // Chordal initialization
    SparseMatrix B1, B2, B3; 
    constructBMatrices(measurements, B1, B2, B3);
    Matrix Rinit = chordalInitialization(problem->dimension(), B3);
    Matrix tinit = recoverTranslations(B1, B2, Rinit);
    Y.resize(r, n*(d+1));
    Y.setZero();
    for (size_t i=0; i<n; i++)
    {
        Y.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Y.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
    }
    
    RGDMaster master(problem, Y);
    master.solve(8);

    exit(0);
}
