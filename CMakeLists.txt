cmake_minimum_required(VERSION 2.8.11)
project(UKF)

set(CMAKE_CXX_STANDARD 11)

add_library(ukf ukf.cc)
add_library(utils utils.cc)
add_library(types types.cc)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# add_executable(run_filter main.cc)
# target_link_libraries(run_filter ukf types)
# 
add_executable(run_util_tests tests/utils_tests.cc)
target_link_libraries(run_util_tests utils types)

add_executable(run_sensor_tests tests/sensor_tests.cc)
target_link_libraries(run_sensor_tests utils types)
