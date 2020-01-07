#include "SESync.h"
#include "SESync_utils.h"

using namespace std;
using namespace SESync;

int main(int argc, char** argv)
{
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " [input .g2o file]" << endl;
        exit(1);
    }
    AlgType mode;
    if (argc >= 3)
    {
      string modeString(argv[2]);
      if (modeString == "StiefelSync")
        {
        mode = AlgType::StiefelSync;
        }
      else if (modeString == "CartanSync")
        {
        mode = AlgType::CartanSync;
        }
      else
        {
        cout << "Unknown algorithm " << modeString << endl;
        exit(1);
        }
    }
    else
      mode = AlgType::CartanSync; // default mode

    size_t num_poses;
    vector<SESync::RelativePoseMeasurement> measurements = read_g2o_file(argv[1], num_poses);
    cout << "Loaded " << measurements.size() << " measurements between "
         << num_poses << " poses from file " << argv[1] << endl
         << endl;

    SESyncOpts opts;
    opts.verbose = true; // Print output to stdout
    opts.eig_comp_tol = 1e-6; // 1e-10
    opts.min_eig_num_tol = 1e-3; // this is the value used in Matlab version

    /** call actual solver */
    SESyncResult results = SESync::SESync(measurements, mode, opts);

}
