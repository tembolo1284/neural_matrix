cmake_minimum_required(VERSION 3.10)
project(neural_networks C)

# First, build the matrix library
add_subdirectory(${CMAKE_SOURCE_DIR}/../matrix_lib matrix_lib_build)

# Create the neural networks library
add_library(neural_networks
    src/layer.c
    src/dense_layer.c
    src/activation.c
    src/loss.c
    src/optimization.c
)

target_include_directories(neural_networks
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link against matrix_lib
target_link_libraries(neural_networks
    PRIVATE
        matrix
)

# Enable testing
enable_testing()

# Find Criterion library directly
find_library(CRITERION_LIB NAMES criterion)
if(NOT CRITERION_LIB)
    message(FATAL_ERROR "Criterion library not found. Please install criterion.")
endif()

# Create test executables
set(TEST_SOURCES
    tests/layer_test.c
    tests/dense_layer_test.c
    tests/activation_test.c
    tests/loss_test.c
    tests/optimization_test.c
)

# Create a test executable for each test file
foreach(test_source ${TEST_SOURCES})
    # Get the filename without extension
    get_filename_component(test_name ${test_source} NAME_WE)
    
    # Create the test executable
    add_executable(${test_name} ${test_source})
    
    # Link against neural_networks library and Criterion
    target_link_libraries(${test_name}
        PRIVATE
            neural_networks
            matrix
            ${CRITERION_LIB}
            m  # Math library
    )
    
    # Add the test to CTest
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Set compile options
target_compile_options(neural_networks
    PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
)

# Add test coverage if enabled
option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
if(ENABLE_TEST_COVERAGE)
    target_compile_options(neural_networks
        PRIVATE
            --coverage
            -O0
            -g
    )
    target_link_libraries(neural_networks
        PRIVATE
            gcov
    )
endif()
