cmake_minimum_required(VERSION 3.5)

project(roptlib-download NONE)

include(ExternalProject)
ExternalProject_Add(roptlib
        GIT_REPOSITORY    https://github.com/yuluntian/ROPTLIB.git
        GIT_TAG           feature/cmake
        SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlib-src"
        BINARY_DIR        "${CMAKE_CURRENT_BINARY_DIR}/roptlib-build"
        CONFIGURE_COMMAND ""
        BUILD_COMMAND     ""
        INSTALL_COMMAND   ""
        TEST_COMMAND      "")