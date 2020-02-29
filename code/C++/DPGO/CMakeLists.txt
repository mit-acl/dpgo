# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

include(GNUInstallDirs)
set(DPGO_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)


message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

set(EXTERNAL_INCLUDES
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
	${EXTERNAL_INCLUDES}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/eigen>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/ROPTLIB>
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
	# INSTALL
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/eigen>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/DPGO>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/CpxNStQOrth>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/EucPositive>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/Euclidean>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/Grassmann>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/L2Sphere>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/LowRank>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/Oblique>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/OrthGroup>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/SPDManifold>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/Sphere>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Manifolds/Stiefel>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Others>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Others/SparseBLAS>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Problems>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/Solvers>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/cwrapper>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/cwrapper/blas>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/roptlib/cwrapper/lapack>	
	)

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
include(CMakePackageConfigHelpers)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/cmake/DPGO)

install(TARGETS DPGO
  EXPORT 	DPGOTargets	
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

install(EXPORT DPGOTargets
  FILE DPGOTargets.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)


write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/DPGOConfigVersion.cmake
  VERSION 1.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(${CMAKE_CURRENT_LIST_DIR}/cmake/DPGOConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/DPGOConfig.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)

# install the configuration file
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/DPGOConfig.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include/
  DESTINATION ${DPGO_INSTALL_INCLUDEDIR}/DPGO
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/ROPTLIB/
  DESTINATION ${DPGO_INSTALL_INCLUDEDIR}/roptlib
  FILES_MATCHING PATTERN "*.h"
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/eigen/
  DESTINATION ${DPGO_INSTALL_INCLUDEDIR}/eigen
)


export(TARGETS DPGO FILE DPGOTargets.cmake)