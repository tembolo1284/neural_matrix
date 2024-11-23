cmake_minimum_required(VERSION 3.10)

# Define the matrix library
add_library(matrix STATIC
    src/matrix.c
)

# Include directory for matrix
target_include_directories(matrix
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link math library
target_link_libraries(matrix PUBLIC m)

# Compiler options
target_compile_options(matrix
    PRIVATE
        -Wall -Wextra -Werror -Wpedantic
)

# Add main executable for matrix_lib
add_executable(matrix_main main.c)
target_link_libraries(matrix_main PRIVATE matrix)

# Build tests if enabled
if(BUILD_TESTS)
    find_library(CRITERION_LIBRARY criterion REQUIRED)

    # Function to add tests
    function(add_matrix_test test_name)
        add_executable(${test_name} tests/${test_name}.c)
        target_link_libraries(${test_name} PRIVATE matrix ${CRITERION_LIBRARY} m)
        add_test(NAME ${test_name} COMMAND ${test_name})
    endfunction()

    # Add individual tests
    add_matrix_test(test_matrix_init)
    add_matrix_test(test_matrix_math)
    add_matrix_test(test_matrix_operations)
endif()

