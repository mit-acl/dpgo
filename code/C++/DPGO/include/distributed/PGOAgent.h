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

    // Implement a dictionary for easy access of pose value by PoseID
    typedef map<PoseID, Matrix, std::less<PoseID>, 
        Eigen::aligned_allocator<std::pair<PoseID, Matrix>>> PoseDict;

    // Implement a dictionary for easy access of measurement by PoseID
    typedef map<PoseID, RelativeSEMeasurement, std::less<PoseID>> MeasurementDict;

  public:
    // Initialize with an empty pose graph
    PGOAgent(unsigned ID, unsigned dIn, unsigned rIn);

    ~PGOAgent();

    /** Helper function to reset the internal solution
        In deployment, should not use this
     */
    void setY(const Matrix& Yin){Y = Yin;}

    void addOdometry(const RelativeSEMeasurement& factor);

    void addPrivateLoopClosure(const RelativeSEMeasurement& factor);

    void addSharedLoopClosure(const RelativeSEMeasurement& factor);

    /** Store the pose of a neighboring robot who shares loop closure with this robot
        TODO: if necessary (based on the cluster), realign the local frame of this robot to match the neighbor's
        and update the cluster that this robot belongs to 
    */
    void updateSharedPose(unsigned neighborCluster, unsigned neighborID, unsigned neighborPose, const Matrix& var);

    /**
    Return trajectory estimate of this robot in local frame, with its first pose set to identity   
    */
    Matrix getTrajectoryInLocalFrame(); 

    /**
    Return trajectory estimate of this robot in global frame, with the first pose of robot 0 set to identity   
    */
    Matrix getTrajectoryInGlobalFrame();



    

  private:
    unsigned mID; // The unique ID associated to this robot
    unsigned mCluster; // The cluster that this robot belongs to 
    unsigned d;
    unsigned r;
    unsigned n;

    Matrix Y;
    
    // used by getTrajectoryInGlobalFrame
    Matrix globalAnchor; 

    vector<RelativeSEMeasurement> odometry;
    vector<RelativeSEMeasurement> privateLoopClosures;

    // Stores ID of other robots that share loop closures
    vector<unsigned> neighborAgents;

    // This dictionary stores poses owned by other robots that is connected to this robot by loop closure
    PoseDict sharedPoseDict;
    
    // This dictionary stores shared loop closure measurements
    MeasurementDict sharedMeasurementDict;

    // Implement locking to synchronize read & write of trajectory estimate
    mutex mTrajectoryMutex;

    // Implement locking to synchronize read & write of shared poses from neighbors
    mutex mSharedPosesMutex;

  };

} 




#endif