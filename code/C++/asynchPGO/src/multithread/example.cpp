#include <iostream>
#include "SESync.h"
#include "SESync_utils.h"
#include "CartanSyncProblem.h"
#include "QuadraticProblem.h"
#include "multithread/RGDMaster.h"

using namespace std;
using namespace AsynchPGO;

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
    cout << "Number of poses: " << n
    << ". Dimension: " << d
    << ". Relaxation rank: " << r << endl;
    cout << "Q size: " << ConLapT.rows() << "," << ConLapT.cols() << endl;
    cout << "G size: " << G.rows() << "," << G.cols() << endl;
    QuadraticProblem* problem = new QuadraticProblem(n,d,r,ConLapT,G);

    RGDMaster master;
    master.setProblem(problem);
    
    // master.solve(10);

    exit(0);
}
