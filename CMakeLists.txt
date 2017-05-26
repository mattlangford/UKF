cmake_minimum_required(VERSION 2.8.11)
project(UKF)

set(CMAKE_CXX_STANDARD 11)

add_library(ukf ukf.cc)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

add_executable(run_filter main.cc)
target_link_libraries(run_filter ukf)

add_executable(run_tests tests/utils_tests.cc)

add_executable(run_sensor_tests tests/sensor_tests.cc)
