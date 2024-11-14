cmake_minimum_required(VERSION 3.10)
project(neural_matrix C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Add subdirectories
add_subdirectory(neural_networks)
add_subdirectory(examples)

# Testing options
option(BUILD_TESTS "Build tests" ON)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(neural_networks/tests)
endif()
