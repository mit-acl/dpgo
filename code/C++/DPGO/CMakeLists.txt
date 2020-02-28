# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 3.5)

project(DPGO VERSION 1.0)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

set(EXTERNAL_INCLUDES
	${EIGEN3_INCLUDE_DIR}
	${CHOLMOD_INCLUDES}
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

target_include_directories(DPGO PUBLIC 
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/CpxNStQOrth>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/EucPositive>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/Euclidean>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/Grassmann>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/L2Sphere>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/LowRank>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/Oblique>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/OrthGroup>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/SPDManifold>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/Sphere>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Manifolds/Stiefel>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Others>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Others/SparseBLAS>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Problems>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/Solvers>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/cwrapper>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/cwrapper/blas>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB/cwrapper/lapack>	
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/spectra/include>	
	${EXTERNAL_INCLUDES})

target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} )

# Build tests
add_executable(test-optimization-thread
	tests/testOptimizationThread.cpp)
target_link_libraries(test-optimization-thread DPGO)

# Build multithread PGO example
add_executable(multithread-example
	examples/MultithreadExample.cpp)
target_link_libraries(multithread-example DPGO)

# Build Distributed PGO example
add_executable(distributed-example
	examples/DistributedExample.cpp)
target_link_libraries(distributed-example DPGO)

add_executable(distributed-example2
	examples/DistributedExample2.cpp)
target_link_libraries(distributed-example2 DPGO)


# Install
include(GNUInstallDirs)

install(TARGETS DPGO
	EXPORT 	DPGOTargets	
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    	ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )

install(EXPORT DPGOTargets
  FILE DPGOTargets.cmake
  DESTINATION lib/cmake/DPGO
)

