#################################################################################
# This is CMakeLists for build BRDFRead.cpp with boost.Python, Boost.Numpy.
#
# Copyright (C) 2017 Hiroaki Santo
#################################################################################

cmake_minimum_required(VERSION 2.8)
project(merl_brdf_database)
set(CMAKE_BUILD_TYPE Release)
find_package(Boost COMPONENTS python REQUIRED)
find_package(PythonLibs REQUIRED)
include_directories(${Boost_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIRS})

add_library(BRDFRead SHARED BRDFRead.cpp)
target_link_libraries(BRDFRead ${Boost_LIBRARIES} ${PYTHON_LIBRARY} boost_numpy)
set_target_properties(BRDFRead PROPERTIES PREFIX "")