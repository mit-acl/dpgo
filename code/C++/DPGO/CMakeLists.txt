# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

# message(STATUS "EIGEN3_INCLUDE_DIR: " ${EIGEN3_INCLUDE_DIR})
# message(STATUS "ROPTLIB_INCLUDE_DIRECTORIES: " ${ROPTLIB_INCLUDE_DIRECTORIES})

include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${ROPTLIB_INCLUDE_DIRECTORIES})

add_library(DPGO SHARED
	src/manifold/LiftedSEManifold.cpp
	src/manifold/LiftedSEVariable.cpp
	src/manifold/LiftedSEVector.cpp
	src/multithread/RGDMaster.cpp
	src/multithread/RGDWorker.cpp
	src/QuadraticProblem.cpp
	)
target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib)

# Build multithread PGO example
add_executable(multithread-example
	src/examples/MultithreadExample.cpp)
target_link_libraries(multithread-example ${CMAKE_THREAD_LIBS_INIT} SESync DPGO)

