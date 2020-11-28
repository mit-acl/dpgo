# PROJECT CONFIGURATION
cmake_minimum_required(VERSION 2.8)

project(DPGO)

message(STATUS "CXX compiler version: " ${CMAKE_CXX_COMPILER_VERSION})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pthread -O3")

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
	src/PGOAgent.cpp
	src/QuadraticProblem.cpp
	src/QuadraticOptimizer.cpp
	src/DPGO_utils.cpp
	src/PGOLogger.cpp)

target_include_directories(DPGO PUBLIC
	${EXTERNAL_INCLUDES}
	$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/eigen>
	# INSTALL
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/eigen>
	$<INSTALL_INTERFACE:${DPGO_INSTALL_INCLUDEDIR}/DPGO>
	)

target_link_libraries(DPGO ${CMAKE_THREAD_LIBS_INIT} roptlib ${CHOLMOD_LIBRARIES} ${SPQR_LIBRARIES} )

# Build Distributed PGO example
add_executable(multi-robot-example
		examples/MultiRobotExample.cpp)
target_link_libraries(multi-robot-example DPGO)

add_executable(chordal-initialization-example
		examples/ChordalInitializationExample.cpp)
target_link_libraries(chordal-initialization-example DPGO)

add_executable(single-robot-example
	examples/SingleRobotExample.cpp)
target_link_libraries(single-robot-example DPGO)

############################### TESTS ##########################################
### Add testing
option(BUILD_DPGO_TESTS "Build tests" ON)
if(BUILD_DPGO_TESTS)

	configure_file(${CMAKE_CURRENT_SOURCE_DIR}/cmake/gtest.cmake googletest-download/CMakeLists.txt)
	execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
	endif()
	execute_process(COMMAND ${CMAKE_COMMAND} --build .
	  RESULT_VARIABLE result
	  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
	if(result)
	  message(FATAL_ERROR "Build step for googletest failed: ${result}")
	endif()

	# Prevent overriding the parent project's compiler/linker
	# settings on Windows
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

	# Add googletest directly to our build. This defines
	# the gtest and gtest_main targets.
	add_subdirectory("${CMAKE_BINARY_DIR}/external/googletest-src"
                     "${CMAKE_BINARY_DIR}/external/googletest-build"
                     EXCLUDE_FROM_ALL)

	# The gtest/gtest_main targets carry header search path
	# dependencies automatically when using CMake 2.8.11 or
	# later. Otherwise we have to add them here ourselves.
	if (CMAKE_VERSION VERSION_LESS 2.8.11)
	  include_directories("${gtest_SOURCE_DIR}/include")
	endif()

	add_executable(
		testDPGO
		tests/testUtils.cpp
		tests/testConstruction.cpp
		tests/testLineGraph.cpp
		tests/testTriangleGraph.cpp
		tests/testOptimizationThread.cpp
		)
	target_include_directories(testDPGO PUBLIC
		${EXTERNAL_INCLUDES}
		${CMAKE_CURRENT_SOURCE_DIR}/include>
		${CMAKE_SOURCE_DIR}/eigen>
	)
	target_link_libraries(
		testDPGO
		gtest_main
		DPGO
		)
	add_test(
		NAME testDPGO
		COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests
		)
endif(BUILD_DPGO_TESTS)

############################### INSTALL ##########################################
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

############################### UNINSTALL ##########################################
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()
