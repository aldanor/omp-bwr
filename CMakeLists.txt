cmake_minimum_required(VERSION 3.12)
project(omp_bwr)

set(CMAKE_CXX_STANDARD 14)

find_package(OPENMP REQUIRED)

add_executable(test_omp_bwr src/test.cpp)
target_link_libraries(test_omp_bwr PRIVATE OpenMP::OpenMP_CXX)
