# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)
set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel.")


project(SE-Sync C CXX)

set (CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
add_compile_options(-march=native) # Enable faster instruction sets

set(SESync_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(SESync_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(SESync_EXAMPLES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples)

# Expose the include directories for this project
set(SESYNC_INCLUDES ${SESync_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR} ${SPQR_INCLUDES} ${CHOLMOD_INCLUDES} ${SPECTRA_INCLUDE_DIR}  ${ROPTLIB_INCLUDE_DIRECTORIES} CACHE INTERNAL "")

# Get the set of SE-Sync header and source files
set(SESync_HDRS
${SESync_INCLUDE_DIR}/CartanSyncManifold.h
${SESync_INCLUDE_DIR}/CartanSyncVariable.h
${SESync_INCLUDE_DIR}/CartanSyncVector.h
${SESync_INCLUDE_DIR}/RelativePoseMeasurement.h
${SESync_INCLUDE_DIR}/SESync_types.h
${SESync_INCLUDE_DIR}/SESync_utils.h
${SESync_INCLUDE_DIR}/SESyncProblem.h
${SESync_INCLUDE_DIR}/StiefelSyncProblem.h
${SESync_INCLUDE_DIR}/CartanSyncProblem.h
${SESync_INCLUDE_DIR}/SESync.h
)

set(SESync_SRCS
${SESync_SOURCE_DIR}/CartanSyncManifold.cpp
${SESync_SOURCE_DIR}/CartanSyncVariable.cpp
${SESync_SOURCE_DIR}/CartanSyncVector.cpp
${SESync_SOURCE_DIR}/SESync_utils.cpp
${SESync_SOURCE_DIR}/StiefelSyncProblem.cpp
${SESync_SOURCE_DIR}/CartanSyncProblem.cpp
${SESync_SOURCE_DIR}/SESync.cpp
${SESync_SOURCE_DIR}/CartanSync.cpp
)

message(STATUS "Building in ${CMAKE_BUILD_TYPE} mode\n")

# Build the SE-Sync library
add_library(SESync SHARED ${SESync_HDRS} ${SESync_SRCS})
target_include_directories(SESync PUBLIC ${SESYNC_INCLUDES})
target_link_libraries(SESync roptlib ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${M} ${LAPACK})


# Build the example executable
add_executable(se-sync ${SESync_EXAMPLES_DIR}/main.cpp)
target_include_directories(se-sync PUBLIC ${SESYNC_INCLUDES})
target_link_libraries(se-sync SESync ${BLAS_LIBRARIES} ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} ${M} ${LAPACK})
