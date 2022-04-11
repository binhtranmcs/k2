if("x${CMAKE_SOURCE_DIR}" STREQUAL "x${CMAKE_BINARY_DIR}")
  message(FATAL_ERROR "\
In-source build is not a good practice.
Please use:
  mkdir build
  cd build
  cmake ..
to build this project"
  )
endif()

cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_VERBOSE_MAKEFILE on)

# see https://cmake.org/cmake/help/latest/policy/CMP0111.html
#
# This is to suppress the warnings for importing PyTorch.
if(POLICY CMP0111)
  cmake_policy(SET CMP0111 OLD)
endif()
if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()


project(k2 CXX)

set(K2_VERSION "1.14")

# ----------------- Supported build types for K2 project -----------------
set(ALLOWABLE_BUILD_TYPES Debug Release RelWithDebInfo MinSizeRel)
set(DEFAULT_BUILD_TYPE "Debug")
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "${ALLOWABLE_BUILD_TYPES}")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  # CMAKE_CONFIGURATION_TYPES: with config type values from other generators (IDE).
  message(STATUS "No CMAKE_BUILD_TYPE given, default to Debug")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}")
elseif(NOT CMAKE_BUILD_TYPE IN_LIST ALLOWABLE_BUILD_TYPES)
  message(FATAL_ERROR "Invalid build type: ${CMAKE_BUILD_TYPE}, \
    choose one from ${ALLOWABLE_BUILD_TYPES}")
endif()

string(TOUPPER ${CMAKE_BUILD_TYPE} CMAKE_BUILD_TYPE_UPPERCASE)
if("${CMAKE_BUILD_TYPE_UPPERCASE}" STREQUAL "DEBUG")
  # refer to https://docs.nvidia.com/cuda/cuda-memcheck/index.html#compilation-options
  # The two options are to make cuda-memcheck's stack backtrace feature more useful.
  string(APPEND CMAKE_CUDA_FLAGS " --compiler-options -rdynamic --compiler-options -lineinfo")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_SHARED_LIBS "Whether to build shared or static lib" ON)
option(K2_USE_PYTORCH "Whether to build with PyTorch" ON)
option(K2_ENABLE_BENCHMARK "Whether to enable benchmark" ON)
option(K2_WITH_CUDA "Whether to build k2 with CUDA" OFF)

# If K2_WITH_CUDA is ON, then K2_ENABLE_NVTX has a default value ON
# If K2_WITH_CUDA is OFF, then K2_ENABLE_NVTX is set to OFF
include(CMakeDependentOption)
cmake_dependent_option(K2_ENABLE_NVTX "Whether to build with the NVTX library" ON K2_WITH_CUDA OFF)

if(NOT K2_USE_PYTORCH)
  message(FATAL_ERROR "\
    Please set K2_USE_PYTORCH to ON.
    Support for other frameworks will be added later")
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(BUILD_RPATH_USE_ORIGIN TRUE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")

if(UNIX AND NOT APPLE)
  execute_process(COMMAND
    lsb_release -sd
    OUTPUT_VARIABLE K2_OS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
elseif(APPLE)
  execute_process(COMMAND
    sw_vers -productName
    OUTPUT_VARIABLE _product_name
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -productVersion
    OUTPUT_VARIABLE _product_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  execute_process(COMMAND
    sw_vers -buildVersion
    OUTPUT_VARIABLE _build_version
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(K2_OS "${_product_name} ${_product_version} ${_build_version}")
elseif(WIN32)
  execute_process(COMMAND
    wmic os get caption,version
    OUTPUT_VARIABLE K2_OS_TWO_LINES
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # Now K2_OS_TWO_LINES contains something like
  #  Caption                          Version
  #  Microsoft Windows 10 Pro         10.0.18362
  string(REPLACE "\n" ";" K2_OS_LIST ${K2_OS_TWO_LINES})
  list(GET K2_OS_LIST 1 K2_OS)
else()
  set(K2_OS "Unknown")
endif()

string(REGEX REPLACE "^\"+|\"+$" "" K2_OS "${K2_OS}")
message(STATUS "K2_OS: ${K2_OS}")

find_package(Git REQUIRED)
execute_process(COMMAND
  "${GIT_EXECUTABLE}" describe --always --abbrev=40
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE K2_GIT_SHA1
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(COMMAND
  "${GIT_EXECUTABLE}" log -1 --format=%ad --date=local
  WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
  OUTPUT_VARIABLE K2_GIT_DATE
  ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE
)

include(CheckIncludeFileCXX)
check_include_file_cxx(cxxabi.h K2_HAVE_CXXABI_H)
check_include_file_cxx(execinfo.h K2_HAVE_EXECINFO_H)

include(CheckCXXCompilerFlag)
if(NOT WIN32)
  check_cxx_compiler_flag("-std=c++14" K2_COMPILER_SUPPORTS_CXX14)
else()
  # windows x86 or x86_64
  check_cxx_compiler_flag("/std:c++14" K2_COMPILER_SUPPORTS_CXX14)
endif()
if(NOT K2_COMPILER_SUPPORTS_CXX14)
  message(FATAL_ERROR "
    k2 requires a compiler supporting at least C++14.
    If you are using GCC, please upgrade it to at least version 5.0.
    If you are using Clang, please upgrade it to at least version 3.4.")
endif()

# ========= Settings for CUB begin =========
# the following settings are modified from cub/CMakeLists.txt
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_CXX_EXTENSIONS OFF)

message(STATUS "C++ Standard version: ${CMAKE_CXX_STANDARD}")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(Valgrind)
if(Valgrind_FOUND)
  find_program(MEMORYCHECK_COMMAND NAMES ${Valgrind_EXECUTABLE})
  set(MEMORYCHECK_COMMAND_OPTIONS "--suppressions=${CMAKE_SOURCE_DIR}/scripts/valgrind.supp --leak-check=full")
  include(Dart)
  message(STATUS "To check memory, run `ctest -R <NAME> -D ExperimentalMemCheck`")
endif()
enable_testing()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#if(K2_USE_PYTORCH)
  #add_definitions(-DK2_USE_PYTORCH)
  #add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)
  #include(torch)
#endif()

set(build_DIR ${CMAKE_SOURCE_DIR}/../../../build)
# Pytorch
add_definitions(-DK2_USE_PYTORCH)
add_definitions(-DTORCH_API_INCLUDE_EXTENSION_H)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

find_library(PYTORCH_LIBRARIES pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)

find_library(FBJNI_LIBRARY fbjni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH
)

list(APPEND TORCH_LIBRARIES ${PYTORCH_LIBRARIES})
list(APPEND TORCH_LIBRARIES ${FBJNI_LIBRARY})

include_directories(
  ${PYTORCH_INCLUDE_DIRS}
  ${PYTORCH_INCLUDE_DIRS}/torch/csrc/api/include
)
set(K2_TORCH_VERSION_MAJOR 1)
set(K2_TORCH_VERSION_MINOR 10)

link_libraries(log libc++_shared.so android)
include(googletest)
include(sentencepiece)
include(kaldifeat)

if(NOT K2_WITH_CUDA AND NOT WIN32)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-unused-variable")
endif()

if(NOT WIN32)
  string(APPEND CMAKE_CXX_FLAGS " -Wno-strict-overflow")
endif()

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

add_subdirectory(k2)

