# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")

find_package(Threads REQUIRED)

message(STATUS "CMAKE_MODULE_PATH: " ${CMAKE_MODULE_PATH})

set(EXTERNAL_INCLUDES
	${CHOLMOD_INCLUDES}
	${EIGEN3_INCLUDE_DIR}
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
	src/DPGO_robust.cpp
	)

target_include_directories(DPGO PUBLIC 
	${EXTERNAL_INCLUDES}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/eigen>
	# INSTALL
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/eigen>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/DPGO>
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
install(DIRECTORY ${CMAKE_SOURCE_DIR}/eigen/
  DESTINATION ${DPGO_INSTALL_INCLUDEDIR}/eigen
)


export(TARGETS DPGO FILE DPGOTargets.cmake)
