cmake_minimum_required(VERSION 3.10)
project(neural_networks C)

# Add debug flags and disable optimizations for easier debugging
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O0")

# Locate the Criterion testing library
find_library(CRITERION_LIBRARY criterion REQUIRED)
if(NOT CRITERION_LIBRARY)
    message(FATAL_ERROR "Criterion library not found. Please install the criterion-dev package")
endif()

# Add the matrix library subdirectory
add_subdirectory(../matrix_lib matrix_lib_build)

# Create the neural networks library
add_library(neural_networks
    src/layer.c
    src/dense_layer.c
    src/activation.c
    src/optimization.c
    src/loss.c
    src/log.c
)

# Include directories for neural_networks
target_include_directories(neural_networks
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link the neural_networks library with matrix
target_link_libraries(neural_networks
    PUBLIC
        matrix
)

# Compiler options for neural_networks
target_compile_options(neural_networks
    PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
)

# Function to add test executable and link dependencies
function(add_neural_test test_name)
    add_executable(${test_name} tests/${test_name}.c)
    target_link_libraries(${test_name}
        PRIVATE
            neural_networks
            matrix
            ${CRITERION_LIBRARY}
            m
    )
    target_compile_options(${test_name}
        PRIVATE
            -g  # Add debug symbols
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Add tests if the tests directory exists
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    enable_testing()
    add_neural_test(layer_test)
    add_neural_test(dense_layer_test)
    add_neural_test(activation_test)
    add_neural_test(optimization_test)
    add_neural_test(loss_test)
endif()

# Add the main executable
add_executable(neural_networks_main src/main.c)
target_link_libraries(neural_networks_main
    PRIVATE
        neural_networks
        matrix
        m
)
target_compile_options(neural_networks_main
    PRIVATE
        -Wall
        -Wextra
        -Werror
        -Wpedantic
        -g
)
