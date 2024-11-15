#include "activation.h"
#include <criterion/criterion.h>
#include <stdio.h>

void activation_test_impl(activation_type type, unsigned int dim) {
    fprintf(stderr, "Starting activation test for type=%d, dim=%u...\n", type, dim);

    // Step 1: Create activation layer
    fprintf(stderr, "Creating activation layer...\n");
    layer* l = activation_layer_new(dim, type);
    cr_assert_not_null(l, "Failed to create activation layer");

    // Step 2: Create input matrix
    fprintf(stderr, "Creating input matrix...\n");
    matrix* input = matrix_new(dim, 1, sizeof(double));
    for (unsigned int i = 0; i < dim; i++) {
        matrix_set(input, i, 0, (double)i / dim - 0.5);  // Example data
    }

    // Step 3: Perform forward pass
    fprintf(stderr, "Performing forward pass...\n");
    matrix* output = activation_forward(l, input);
    cr_assert_not_null(output, "Forward pass failed");

    // Step 4: Perform backward pass
    fprintf(stderr, "Creating gradient matrix...\n");
    matrix* gradient = matrix_new(dim, 1, sizeof(double));
    for (unsigned int i = 0; i < dim; i++) {
        matrix_set(gradient, i, 0, 1.0);  // Example gradient
    }

    fprintf(stderr, "Performing backward pass...\n");
    matrix* input_gradient = activation_backward(l, gradient);
    cr_assert_not_null(input_gradient, "Backward pass failed");

    // Step 5: Free resources
    fprintf(stderr, "Freeing resources...\n");
    matrix_free(input);
    matrix_free(output);
    matrix_free(gradient);
    matrix_free(input_gradient);
    activation_free(l);

    fprintf(stderr, "Completed activation test for type=%d, dim=%u\n", type, dim);
}

Test(activation_relu, forward_and_backward) {
    activation_test_impl(ACTIVATION_RELU, 4);
}

Test(activation_sigmoid, forward_and_backward) {
    activation_test_impl(ACTIVATION_SIGMOID, 4);
}

Test(activation_tanh, forward_and_backward) {
    activation_test_impl(ACTIVATION_TANH, 4);
}

Test(activation_softmax, forward_and_backward) {
    activation_test_impl(ACTIVATION_SOFTMAX, 4);
}

