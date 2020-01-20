#ifndef PGOAGENT_H
#define PGOAGENT_H

#include <vector>
#include <set>
#include <map>
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
  // In distributed PGO, each pose is uniquely determined by the robot ID and pose ID
  typedef pair<unsigned, unsigned> PoseID;

  // Implement a dictionary for easy access of pose value by PoseID
  typedef map<PoseID, Matrix, std::less<PoseID>, 
          Eigen::aligned_allocator<std::pair<PoseID, Matrix>>> PoseDict;

  class PGOAgent{
  public:
    // Initialize with an empty pose graph
    PGOAgent(unsigned ID, unsigned dIn, unsigned rIn);
    PGOAgent(unsigned ID, unsigned dIn, unsigned rIn, bool v);
    ~PGOAgent();

    /** Helper function to reset the internal solution
        In deployment, should not use this
     */
    void setY(const Matrix& Yin)
    {
        Y = Yin;
        assert(Y.cols() == n * (d+1));
        assert(Y.rows() == r);
    }
    Matrix getY(){return Y;};

    /**
    Add an odometric measurement of this robot.
    This function automatically initialize the new pose, by propagating odometry
    */
    void addOdometry(const RelativeSEMeasurement& factor);

    /**
    Add a private loop closure of this robot
    */
    void addPrivateLoopClosure(const RelativeSEMeasurement& factor);


    /**
    Add a shared loop closure between this robot and another
    */
    void addSharedLoopClosure(const RelativeSEMeasurement& factor);

    /** 
    Store the pose of a neighboring robot who shares loop closure with this robot
    TODO: if necessary (based on the cluster), realign the local frame of this robot to match the neighbor's
    and update the cluster that this robot belongs to 
    */
    void updateNeighborPose(unsigned neighborCluster, unsigned neighborID, unsigned neighborPose, const Matrix& var);

    /** 
    Optimize pose graph by a single iteration. 
    This process use both private and shared factors (communication required for the latter)
    */
    void optimize();

    /**
    Optimize pose graph by periodically calling optimize()
    */
    void runOptimizationLoop();

    /**
    Return trajectory estimate of this robot in local frame, with its first pose set to identity   
    */
    Matrix getTrajectoryInLocalFrame(); 

    /**
    Return trajectory estimate of this robot in global frame, with the first pose of robot 0 set to identity   
    */
    Matrix getTrajectoryInGlobalFrame();


    /**
    Return a map of shared poses of this robot, that need to be sent to others
    */
    PoseDict getSharedPoses();



    

  private:
    
    // The unique ID associated to this robot
    unsigned mID; 

    // The cluster that this robot belongs to 
    unsigned mCluster; 
    
    // Dimension
    unsigned d;

    // Relaxed rank in Riemanian optimization problem
    unsigned r;
    
    // Number of poses
    unsigned n;

    // Verbose flag
    bool verbose;

    // Solution before rounding
    Matrix Y;
    
    // used by getTrajectoryInGlobalFrame
    Matrix globalAnchor; 

    // Store odometric measurement of this robot
    vector<RelativeSEMeasurement> odometry;

    // Store private loop closures of this robot
    vector<RelativeSEMeasurement> privateLoopClosures;

    // This dictionary stores poses owned by other robots that is connected to this robot by loop closure
    PoseDict neighborPoseDict;

    // Store the set of public poses that need to be sent to other robots
    set<PoseID> mSharedPoses;

    // Store the set of public poses needed from other robots
    set<PoseID> neighborSharedPoses;
    
    // This dictionary stores shared loop closure measurements
    vector<RelativeSEMeasurement> sharedLoopClosures;

    // Implement locking to synchronize read & write of trajectory estimate
    mutex mPosesMutex;

    // Implement locking to synchronize read & write of shared poses from neighbors
    mutex mNeighborPosesMutex;

    // Implement locking on measurements
    mutex mMeasurementsMutex;

    /** Compute the cost matrices that define the local PGO problem
        f(X) = 0.5<Q, XtX> + <X, G>
    */
    void constructCostMatrices(const vector<RelativeSEMeasurement>& privateMeasurements,
            const vector<RelativeSEMeasurement>& sharedMeasurements,
            SparseMatrix* Q, 
            SparseMatrix* G);

  };

} 




#endif