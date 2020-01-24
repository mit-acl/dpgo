# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${CHOLMOD_INCLUDES}
	${ROPTLIB_INCLUDE_DIRECTORIES})

add_library(DPGO SHARED
	src/manifold/LiftedSEManifold.cpp
	src/manifold/LiftedSEVariable.cpp
	src/manifold/LiftedSEVector.cpp
	src/multithread/RGDMaster.cpp
	src/multithread/RGDWorker.cpp
	src/distributed/PGOAgent.cpp
	src/QuadraticProblem.cpp
	src/QuadraticOptimizer.cpp
	src/DPGO_utils.cpp
	)
target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib ${CHOLMOD_LIBRARIES})

# Build tests
add_executable(test-optimization-thread
	tests/testOptimizationThread.cpp)
target_link_libraries(test-optimization-thread ${CMAKE_THREAD_LIBS_INIT} SESync DPGO)

# Build multithread PGO example
add_executable(multithread-example
	examples/MultithreadExample.cpp)
target_link_libraries(multithread-example ${CMAKE_THREAD_LIBS_INIT} SESync DPGO)

# Build Distributed PGO example
add_executable(distributed-example
	examples/DistributedExample.cpp)
target_link_libraries(distributed-example ${CMAKE_THREAD_LIBS_INIT} SESync DPGO)

add_executable(distributed-simulation
	examples/DistributedSimulation.cpp)
target_link_libraries(distributed-simulation ${CMAKE_THREAD_LIBS_INIT} SESync DPGO)
