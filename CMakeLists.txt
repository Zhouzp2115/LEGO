cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(PPML_CPP)

#Relese
set(CMAKE_BUILD_TYPE "Release")

#torch lib
set(CMAKE_PREFIX_PATH "./libtorch-1.2")
find_package(Torch REQUIRED)

#emp-toolkit lib
find_path(CMAKE_FOLDER NAMES cmake/emp-tool-config.cmake)
include(${CMAKE_FOLDER}/cmake/common.cmake)
include(${CMAKE_FOLDER}/cmake/source_of_randomness.cmake)
include(${CMAKE_FOLDER}/cmake/threading.cmake)

find_package(emp-ot REQUIRED)
include_directories(${EMP-OT_INCLUDE_DIRS})

#sub dir
add_subdirectory(${PROJECT_SOURCE_DIR}/src)  

