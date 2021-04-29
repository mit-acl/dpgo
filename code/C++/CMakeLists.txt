# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")

project(DistributedPGO C CXX)

set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
add_compile_options(-march=native) # Enable faster instruction sets

include(GNUInstallDirs)
set(DPGO_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR}/DPGO)

# Directory for built libraries
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/lib CACHE PATH "The directory in which to place libraries built by this project")
# Directory for built executables
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_BINARY_DIR}/bin CACHE PATH "The directory in which to place executables built by this project")

# BUILD CONFIGURATIONS
option(CMAKE_VERBOSE_MAKEFILE "Generate verbose makefiles?" OFF)

# Add the .cmake files that ship with Eigen3 to the CMake module path (useful for finding other stuff)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/eigen/cmake" CACHE STRING "The CMake module path used for this project")
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

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


