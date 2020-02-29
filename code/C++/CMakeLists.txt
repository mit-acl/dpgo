# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

project(DistributedPGO C CXX)

set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
add_compile_options(-march=native) # Enable faster instruction sets


# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place libraries built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# BUILD CONFIGURATIONS
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose makefiles?" OFF)

set(CODE_PROFILING OFF CACHE BOOL "Turn on code profiling?")
if(${CODE_PROFILING})
message(STATUS "Turning on code profiling for SE-Sync")
endif()

# Add the .cmake files that ship with Eigen3 to the CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/eigen/cmake" CACHE STRING "The CMake module path used for this project")


# SESYNC DEPENDENCIES
# FIND EIGEN3
set( ENV{EIGEN3_ROOT_DIR} ${CMAKE_SOURCE_DIR}/eigen)
find_package(Eigen3 3.3 REQUIRED)
if(EIGEN3_FOUND)
message(STATUS "Found Eigen3 library.")
message(STATUS "Eigen3 include directory: ${EIGEN3_INCLUDE_DIR}")
else()
message(STATUS "Eigen library not found!")
endif()
# FIND ADDITIONAL LIBRARIES
# These next operations make use of the .cmake files shipped with Eigen3 
find_package(SPQR REQUIRED)
find_package(Cholmod REQUIRED)
find_package(BLAS REQUIRED)

# FIND SPECTRA
# Spectra doesn't ship with a .cmake file, but it's header-only, so it suffices to simply determine it's top-level directory
set(SPECTRA_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/spectra/include CACHE PATH "Path to top-level directory of Spectra template library")

# Build ROPTLIB library
add_subdirectory(ROPTLIB)

# Build Distributed PGO library
add_subdirectory(DPGO)


