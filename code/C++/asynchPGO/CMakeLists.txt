# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(AsynchPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

# message(STATUS "PROJECT_NAME: " ${PROJECT_NAME})
# message(STATUS "PROJECT_SOURCE_DIR: " ${PROJECT_SOURCE_DIR})
# message(STATUS "SESync_LIBRARIES: " ${SESync_LIBRARIES})

find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
	include
	${EIGEN3_INCLUDE_DIR}
	${SESync_INCLUDE_DIR})

add_executable(multithread
	src/multithread/example.cpp
	src/multithread/RGDMaster.cpp
	src/multithread/RGDWorker.cpp
	src/QuadraticProblem.cpp)
target_link_libraries(multithread ${CMAKE_THREAD_LIBS_INIT} SESync)