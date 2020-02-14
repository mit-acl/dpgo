# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

set(DPGO_INCLUDES
	${CMAKE_CURRENT_LIST_DIR}/include
	${EIGEN3_INCLUDE_DIR}
	${CHOLMOD_INCLUDES}
	${ROPTLIB_INCLUDE_DIRECTORIES}
	${SPECTRA_INCLUDE_DIR}
	CACHE INTERNAL "")


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
target_include_directories(DPGO PUBLIC ${DPGO_INCLUDES})
target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} )

# Build tests
add_executable(test-optimization-thread
	tests/testOptimizationThread.cpp)
target_include_directories(test-optimization-thread PUBLIC ${DPGO_INCLUDES})
target_link_libraries(test-optimization-thread DPGO)

# Build multithread PGO example
add_executable(multithread-example
	examples/MultithreadExample.cpp)
target_include_directories(multithread-example PUBLIC ${DPGO_INCLUDES})
target_link_libraries(multithread-example DPGO)

# Build Distributed PGO example
add_executable(distributed-example
	examples/DistributedExample.cpp)
target_include_directories(distributed-example PUBLIC ${DPGO_INCLUDES})
target_link_libraries(distributed-example DPGO)

add_executable(distributed-example2
	examples/DistributedExample2.cpp)
target_include_directories(distributed-example2 PUBLIC ${DPGO_INCLUDES})
target_link_libraries(distributed-example2 DPGO)

