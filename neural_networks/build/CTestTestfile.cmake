# CMake generated Testfile for 
# Source directory: /home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks
# Build directory: /home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(layer_test "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build/layer_test")
set_tests_properties(layer_test PROPERTIES  _BACKTRACE_TRIPLES "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;61;add_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;67;add_neural_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;0;")
add_test(dense_layer_test "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build/dense_layer_test")
set_tests_properties(dense_layer_test PROPERTIES  _BACKTRACE_TRIPLES "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;61;add_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;68;add_neural_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;0;")
add_test(activation_test "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build/activation_test")
set_tests_properties(activation_test PROPERTIES  _BACKTRACE_TRIPLES "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;61;add_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;69;add_neural_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;0;")
add_test(optimization_test "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build/optimization_test")
set_tests_properties(optimization_test PROPERTIES  _BACKTRACE_TRIPLES "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;61;add_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;70;add_neural_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;0;")
add_test(loss_test "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/build/loss_test")
set_tests_properties(loss_test PROPERTIES  _BACKTRACE_TRIPLES "/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;61;add_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;71;add_neural_test;/home/tembolo381/Desktop/c-workspace/neural_matrix/neural_networks/CMakeLists.txt;0;")
subdirs("matrix_lib_build")
