cmake_minimum_required(VERSION 3.10)
project(neural_networks LANGUAGES C)

# Define the neural_networks library
add_library(neural_networks STATIC
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
        ${CMAKE_SOURCE_DIR}/matrix_lib/include
)

# Link with matrix library
target_link_libraries(neural_networks PUBLIC matrix)

# Compiler options
target_compile_options(neural_networks
    PRIVATE
        -Wall -Wextra -Werror -Wpedantic
)

# Add main executable
add_executable(neural_networks_main src/main.c)
target_link_libraries(neural_networks_main PRIVATE neural_networks matrix m)

# Add tests if enabled
if(BUILD_TESTS)
    find_library(CRITERION_LIBRARY criterion REQUIRED)

    # Function to add tests
    function(add_neural_test test_name)
        add_executable(${test_name} tests/${test_name}.c)
        target_link_libraries(${test_name} PRIVATE neural_networks matrix ${CRITERION_LIBRARY} m)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    # Add neural_networks tests
    add_neural_test(layer_test)
    add_neural_test(dense_layer_test)
    add_neural_test(activation_test)
    add_neural_test(optimization_test)
    add_neural_test(loss_test)
endif()

