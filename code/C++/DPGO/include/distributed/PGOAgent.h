#ifndef PGOAGENT_H
#define PGOAGENT_H

#include <vector>
#include <thread>
#include <mutex>
#include "manifold/LiftedSEManifold.h"
#include "manifold/LiftedSEVariable.h"
#include "manifold/LiftedSEVector.h"
#include "DPGO_types.h"
#include "RelativeSEMeasurement.h"
#include <Eigen/Dense>
#include "Manifolds/Element.h"
#include "Manifolds/Manifold.h"


using namespace std;

/*Define the namespace*/
namespace DPGO{


  class PGOAgent{

  public:
    // In distributed PGO, each pose is uniquely determined by the robot ID and pose ID
    typedef pair<unsigned, unsigned> PoseID;

    // Implement a dictionary for easy access of pose value by its ID
    typedef map<PoseID, Matrix, std::less<PoseID>, 
        Eigen::aligned_allocator<std::pair<PoseID, Matrix>>> PoseDict;


  public:
    // Initialize with an empty pose graph
    PGOAgent(unsigned ID, unsigned dIn, unsigned rIn);

    ~PGOAgent();

    void addOdometry(const RelativeSEMeasurement& factor);

    void addPrivateLoopClosure(const RelativeSEMeasurement& factor);

    void addSharedFactor(const RelativeSEMeasurement& factor);

    void updateNeighborPose(unsigned agent, unsigned pose, const Matrix& var);

    

  private:
    unsigned mID;
    unsigned d;
    unsigned r;
    unsigned n;

    Matrix Y;
    
    vector<RelativeSEMeasurement> odometry;
    vector<RelativeSEMeasurement> privateLoopClosures;
    vector<RelativeSEMeasurement> sharedLoopClosures; 

    // Stores ID of other robots that share loop closures
    vector<unsigned> neighborAgents;

    // This dictionary stores poses owned by other robots that is connected to this robot by loop closure
    PoseDict cachedNeighborPoses;

    // Implement locking to synchronize read & write of trajectory estimate
    mutex mTrajectoryMutex;

  };

} 




#endif