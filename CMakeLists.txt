cmake_minimum_required(VERSION 3.22)
project(omp_report)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP QUIET)

if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(omp_report
        main.cpp
        funcs.cpp
        funcs.h)

if(OpenMP_CXX_FOUND)
    target_link_libraries(omp_report OpenMP::OpenMP_CXX)
endif()