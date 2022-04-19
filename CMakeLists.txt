cmake_minimum_required(VERSION 3.9.0)
project(CmDockOpenCL VERSION 0.2.0)
project(CmDockOpenCL LANGUAGES CXX)

add_definitions(-DCL_TARGET_OPENCL_VERSION=200)
add_definitions(-DCL_HPP_TARGET_OPENCL_VERSION=200)

add_definitions(-DCL_STRUCT_HOST)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

add_executable(${PROJECT_NAME} ./src/main.cpp)

add_subdirectory(src/host)
target_link_libraries(${PROJECT_NAME} hostLibs)

find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()
target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_C)

target_include_directories (${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (${PROJECT_NAME} ${OpenCL_LIBRARY})
