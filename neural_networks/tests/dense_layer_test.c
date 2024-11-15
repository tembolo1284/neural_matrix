#include "dense_layer.h"
#include <criterion/criterion.h>
#include <stdio.h>

// Test 1: Basic Initialization
Test(dense_layer, init) {
    fprintf(stderr, "Starting test_dense_layer_init...\n");

    unsigned int input_dim = 4;
    unsigned int output_dim = 3;
    double weight_min = -0.5;
    double weight_max = 0.5;

    fprintf(stderr, "Creating dense layer with input_dim=%u, output_dim=%u\n", input_dim, output_dim);

    // Initialize the dense layer
    layer* l = dense_layer_new(input_dim, output_dim, weight_min, weight_max);
    cr_assert_not_null(l, "Layer creation failed");
    fprintf(stderr, "Layer initialized successfully\n");

    // Free the layer
    fprintf(stderr, "Freeing dense layer resources...\n");
    dense_free(l);

    fprintf(stderr, "Completed test_dense_layer_init\n");
}

// Test 2: Forward Propagation
Test(dense_layer, forward_propagation) {
    fprintf(stderr, "Starting test_dense_layer_forward...\n");

    unsigned int input_dim = 2;
    unsigned int output_dim = 2;
    double weight_min = -1.0;
    double weight_max = 1.0;

    layer* l = dense_layer_new(input_dim, output_dim, weight_min, weight_max);
    cr_assert_not_null(l, "Layer creation failed");

    fprintf(stderr, "Creating input matrix...\n");
    matrix* input = matrix_new(input_dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);

    fprintf(stderr, "Performing forward propagation...\n");
    matrix* output = dense_forward(l, input);
    cr_assert_not_null(output, "Forward propagation failed");
    fprintf(stderr, "Forward propagation completed successfully\n");

    fprintf(stderr, "Freeing matrices and layer resources...\n");
    matrix_free(input);
    matrix_free(output);
    dense_free(l);

    fprintf(stderr, "Completed test_dense_layer_forward\n");
}

// Test 3: Backward Propagation
Test(dense_layer, backward_propagation) {
    fprintf(stderr, "Starting test_dense_layer_backward...\n");

    unsigned int input_dim = 2;
    unsigned int output_dim = 2;
    double weight_min = -1.0;
    double weight_max = 1.0;

    layer* l = dense_layer_new(input_dim, output_dim, weight_min, weight_max);
    cr_assert_not_null(l, "Layer creation failed");

    fprintf(stderr, "Creating input matrix and performing forward propagation...\n");
    matrix* input = matrix_new(input_dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);

    matrix* output = dense_forward(l, input);
    cr_assert_not_null(output, "Forward propagation failed");

    fprintf(stderr, "Creating gradient matrix for backward propagation...\n");
    matrix* gradient = matrix_new(output_dim, 1, sizeof(double));
    matrix_set(gradient, 0, 0, 0.5);
    matrix_set(gradient, 1, 0, 0.5);

    fprintf(stderr, "Performing backward propagation...\n");
    matrix* input_gradient = dense_backward(l, gradient);
    cr_assert_not_null(input_gradient, "Backward propagation failed");

    fprintf(stderr, "Freeing matrices and layer resources...\n");
    matrix_free(input);
    matrix_free(output);
    matrix_free(gradient);
    matrix_free(input_gradient);
    dense_free(l);

    fprintf(stderr, "Completed test_dense_layer_backward\n");
}

// Test 4: Parameter Update
Test(dense_layer, parameter_update) {
    fprintf(stderr, "Starting test_dense_layer_update...\n");

    unsigned int input_dim = 2;
    unsigned int output_dim = 2;
    double weight_min = -1.0;
    double weight_max = 1.0;

    layer* l = dense_layer_new(input_dim, output_dim, weight_min, weight_max);
    cr_assert_not_null(l, "Layer creation failed");

    fprintf(stderr, "Creating input matrix and performing forward propagation...\n");
    matrix* input = matrix_new(input_dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);

    matrix* output = dense_forward(l, input);
    cr_assert_not_null(output, "Forward propagation failed");

    fprintf(stderr, "Creating gradient matrix for backward propagation...\n");
    matrix* gradient = matrix_new(output_dim, 1, sizeof(double));
    matrix_set(gradient, 0, 0, 0.5);
    matrix_set(gradient, 1, 0, 0.5);

    matrix* input_gradient = dense_backward(l, gradient);
    cr_assert_not_null(input_gradient, "Backward propagation failed");

    fprintf(stderr, "Updating parameters...\n");
    dense_update(l, 0.01);  // Update parameters with a small learning rate
    fprintf(stderr, "Parameter update completed successfully\n");

    fprintf(stderr, "Freeing matrices and layer resources...\n");
    matrix_free(input);
    matrix_free(output);
    matrix_free(gradient);
    matrix_free(input_gradient);
    dense_free(l);

    fprintf(stderr, "Completed test_dense_layer_update\n");
}

