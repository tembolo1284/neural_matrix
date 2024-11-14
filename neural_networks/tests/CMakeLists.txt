enable_testing()

# Find Criterion package
find_package(Criterion REQUIRED)

# Add test executables
add_executable(test_layer tests/test_layer.c)
add_executable(test_dense_layer tests/test_dense_layer.c)
add_executable(test_activation tests/test_activation.c)
add_executable(test_loss tests/test_loss.c)

# Link libraries
target_link_libraries(test_layer PRIVATE neural_networks Criterion::Criterion)
target_link_libraries(test_dense_layer PRIVATE neural_networks Criterion::Criterion)
target_link_libraries(test_activation PRIVATE neural_networks Criterion::Criterion)
target_link_libraries(test_loss PRIVATE neural_networks Criterion::Criterion)

# Add tests to CTest
add_test(NAME test_layer COMMAND test_layer)
add_test(NAME test_dense_layer COMMAND test_dense_layer)
add_test(NAME test_activation COMMAND test_activation)
add_test(NAME test_loss COMMAND test_loss)
