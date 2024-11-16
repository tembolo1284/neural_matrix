cmake_minimum_required(VERSION 3.10)
project(neural_matrix C)

# Set policies
cmake_policy(SET CMP0079 NEW)

# Set C standard for all projects
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Common compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")

# Testing options
option(BUILD_TESTS "Build all tests" ON)

# Enable testing at root level if tests are enabled
if(BUILD_TESTS)
    enable_testing()
endif()

# Before including matrix_lib, define its target name
set(MATRIX_TARGET_NAME root_matrix CACHE STRING "Name of the matrix library target")

# Add matrix_lib first as it's a dependency
add_subdirectory(matrix_lib ${CMAKE_BINARY_DIR}/matrix_lib)

# Then add neural_networks
add_subdirectory(neural_networks ${CMAKE_BINARY_DIR}/neural_networks)
