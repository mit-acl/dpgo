#include <iostream>
#include <fstream>
#include <cstdlib>
#include <chrono>
#include "QuadraticProblem.h"
#include "distributed/PGOAgent.h"
#include "DPGO_types.h"
#include "DPGO_utils.h"

using namespace std;
using namespace DPGO;


/**
This demo simulates *parallel* distributed pose graph optimization 

Robots optimize local pose graphs concurrently in multiple threads.

The communication is simulated by the master thread running the main() function.

*/

void exchangeSharedPoses(vector<PGOAgent*>& agents){
    for(size_t robot1 = 0; robot1 < agents.size(); ++robot1){
        PoseDict sharedPoses = agents[robot1]->getSharedPoses();
        for(size_t robot2 = 0; robot2 < agents.size(); ++robot2){
            if(robot1 == robot2) continue;
            
            for(auto it = sharedPoses.begin(); it != sharedPoses.end(); ++it){
                PoseID nID = it->first; 
                Matrix var = it->second;
                unsigned agentID = get<0>(nID);
                unsigned localID = get<1>(nID);
                agents[robot2]->updateNeighborPose(0,agentID,localID, var);
            }

        }
    }
}

static void show_usage(std::string name)
{
        cout  << "Usage: " << name << " <option(s)> [.g2o file] \n"
              << "Options:\n"
              << "\t--help \t\t\t Show this help message\n"
              << "\t--robots NUMROBOTS \t Specify number of robots to simulate\n"
              << "\t--rate RATE \t\t Specify update rate for each robot \n"
              << "\t--useRGD STEPSIZE \t Use Riemannian Gradient Descent with specified stepsize; otherwise use RTR (default) \n" 
              << "\t--verbose \t\t Turn on verbose output"
              << std::endl;
}


int main(int argc, char** argv)
{
    cout << "Distributed pose-graph optimization simulation. " << endl;

    /**
    ###########################################
    Parse command line inputs
    ###########################################
    */
    if (argc < 2) {
        show_usage(argv[0]);
        exit(1);
    }

    unsigned r = 5;
    unsigned num_robots = 2;
    double rate = 10; 
    double stepsize = 1e-3;
    bool verbose = false;
    ROPTALG algorithm = ROPTALG::RTR;
    std::string datasetFile;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            show_usage(argv[0]);
            exit(0);
        } 
        else if (arg == "--robots") {
            if (i + 1 < argc) {
                int argInt = atoi(argv[++i]);
                if(argInt <= 0){
                    std::cerr << "robots must be positive." << endl;
                    exit(1);
                }
                num_robots = (unsigned) argInt;
                cout << "Simulating " << num_robots << " robots." << endl;
            } else { 
                cerr << "--robots option requires one argument." << endl;
                exit(1);
            }  
        } 
        else if (arg == "--rate") {
            if (i + 1 < argc) {
                rate = stod(argv[++i]);
                if(rate <= 0){
                    std::cerr << "rate must be positive." << endl;
                    exit(1);
                }
                cout << "Update rate set to " << rate << " Hz." << endl;
            } else { 
                cerr << "--rate option requires one argument." << endl;
                exit(1);
            }  
        } 
        else if (arg == "--useRGD") {
            if (i + 1 < argc) {
                algorithm = ROPTALG::RGD;
                stepsize = stod(argv[++i]);
                if(rate <= 0){
                    std::cerr << "step size must be positive." << endl;
                    exit(1);
                }
                cout << "Using RGD with step size " << stepsize << "." << endl;
            } else { 
                cerr << "--useRGD option requires one argument." << endl;
                exit(1);
            }  
        } 
        else if (arg == "--verbose"){
            verbose = true;
            cout << "Turn on verbose output." << endl; 
        }
        else {
            datasetFile = arg;
        }
    }


    /**
    ###########################################
    Read dataset
    ###########################################
    */
    size_t num_poses;
    vector<RelativeSEMeasurement> dataset = read_g2o_file(datasetFile, num_poses);
    unsigned n = num_poses;
    unsigned d = (!dataset.empty() ? dataset[0].t.size() : 0);
    SparseMatrix ConLapT = constructConnectionLaplacianSE(dataset);
    cout << "Loaded dataset from file " << datasetFile << "." << endl;
    sleep(1);

    /**
    ###########################################
    Initialize multiple PGOAgents
    ###########################################
    */
    unsigned int num_poses_per_robot = n/num_robots;
    if(num_poses_per_robot <= 0){
        cout << "More robots than total number of poses! Decrease the number of robots" << endl;
        exit(1);
    }

    // create mapping from global pose index to local pose index
    map<unsigned, PoseID> PoseMap;
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned) num_robots - 1) endIdx = n;
        for(unsigned idx = startIdx; idx < endIdx; ++idx){
            unsigned localIdx = idx - startIdx; // this is the local ID of this pose
            PoseID pose = make_pair(robot, localIdx);
            PoseMap[idx] = pose;
        }
    }


    PGOAgentParameters options(d,r,algorithm,verbose);
    vector<PGOAgent*> agents;



    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        PGOAgent* ag = new PGOAgent(robot, options);
        ag->setStepsize(stepsize);
        agents.push_back(ag);
    }


    for(size_t k = 0; k < dataset.size(); ++k){
        RelativeSEMeasurement mIn = dataset[k];
        PoseID src = PoseMap[mIn.p1];
        PoseID dst = PoseMap[mIn.p2];

        unsigned srcRobot = get<0>(src);
        unsigned srcIdx = get<1>(src);
        unsigned dstRobot = get<0>(dst);
        unsigned dstIdx = get<1>(dst);

        RelativeSEMeasurement m(srcRobot, dstRobot, srcIdx, dstIdx, mIn.R, mIn.t, mIn.kappa, mIn.tau);

        if (srcRobot == dstRobot){
            // private measurement
            if(srcIdx + 1 == dstIdx){
                // Odometry
                agents[srcRobot]->addOdometry(m);
            }
            else{
                // private loop closure
                agents[srcRobot]->addPrivateLoopClosure(m);
            }
        }else{
            // shared measurement
            agents[srcRobot]->addSharedLoopClosure(m);
            agents[dstRobot]->addSharedLoopClosure(m);
        }

    }

    /**
    ###################################################
    Compute initialization (currently requires SE-Sync)
    ###################################################
    */
    cout << "Initializing..." << endl;

    Matrix Yinit;
    SparseMatrix B1, B2, B3; 
    constructBMatrices(dataset, B1, B2, B3);
    Matrix Rinit = chordalInitialization(d, B3);
    Matrix tinit = recoverTranslations(B1, B2, Rinit);
    Yinit.resize(r, n*(d+1));
    Yinit.setZero();
    for (size_t i=0; i<n; i++)
    {
        Yinit.block(0,i*(d+1),  d,d) = Rinit.block(0,i*d,d,d);
        Yinit.block(0,i*(d+1)+d,d,1) = tinit.block(0,i,d,1);
    }

    
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        unsigned startIdx = robot * num_poses_per_robot;
        unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
        if (robot == (unsigned) num_robots - 1) endIdx = n;

        agents[robot]->setY(Yinit.block(0, startIdx*(d+1), r, (endIdx-startIdx)*(d+1)));
        
    }

    /**
    ###########################################
    Optimize!
    ###########################################
    */
    
    Matrix Yopt = Yinit;
    exchangeSharedPoses(agents);
    
    // Initiate optimization thread for each agent
    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        agents[robot]->startOptimizationLoop(rate);
        usleep(1000);
    }

    auto startTime = std::chrono::high_resolution_clock::now();

    while (true){
        auto counter = std::chrono::high_resolution_clock::now() - startTime;
        double elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(counter).count();
        double elapsedSecond = elapsedMs / 1000.0;

        // Evaluate
        cout 
        << "Time = " << elapsedSecond << " sec | "
        << "cost = " << (Yopt * ConLapT * Yopt.transpose()).trace() << endl;
        
        exchangeSharedPoses(agents);

        for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
            unsigned startIdx = robot * num_poses_per_robot;
            unsigned endIdx = (robot+1) * num_poses_per_robot; // non-inclusive
            if (robot == (unsigned) num_robots - 1) endIdx = n;

            Yopt.block(0, startIdx*(d+1), r, (endIdx-startIdx)*(d+1)) = agents[robot]->getY();
            
        }

        if (elapsedSecond > 20) break;

        usleep(1e5);
    }

    for(unsigned robot = 0; robot < (unsigned) num_robots; ++robot){
        agents[robot]->endOptimizationLoop();
    }

    exit(0);
}