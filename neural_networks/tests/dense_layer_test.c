// tests/dense_layer_test.c
#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/dense_layer.h"

Test(dense_layer_init, create_dense_layer) {
    unsigned int input_dim = 3;
    unsigned int output_dim = 2;
    double weight_min = -1.0;
    double weight_max = 1.0;
    
    layer* l = dense_layer_new(input_dim, output_dim, weight_min, weight_max);
    
    cr_assert_not_null(l, "Dense layer creation failed");
    cr_assert_eq(l->input_dim, input_dim, "Input dimension mismatch");
    cr_assert_eq(l->output_dim, output_dim, "Output dimension mismatch");
    cr_assert_not_null(l->forward, "Forward function should be set");
    cr_assert_not_null(l->backward, "Backward function should be set");
    cr_assert_not_null(l->update, "Update function should be set");
    cr_assert_not_null(l->free, "Free function should be set");
    
    dense_parameters* params = (dense_parameters*)l->parameters;
    cr_assert_not_null(params, "Parameters should not be NULL");
    cr_assert_not_null(params->weights, "Weights should not be NULL");
    cr_assert_not_null(params->bias, "Bias should not be NULL");
    
    cr_assert_eq(params->weights->num_rows, output_dim, "Weight matrix rows mismatch");
    cr_assert_eq(params->weights->num_cols, input_dim, "Weight matrix cols mismatch");
    cr_assert_eq(params->bias->num_rows, output_dim, "Bias vector dimension mismatch");
    cr_assert_eq(params->bias->num_cols, 1, "Bias should be a column vector");
    
    layer_free(l);
}

Test(dense_layer_forward, forward_propagation) {
    unsigned int input_dim = 2;
    unsigned int output_dim = 1;
    layer* l = dense_layer_new(input_dim, output_dim, 0.0, 0.0); // Initialize weights to 0
    
    // Set specific weights and bias for testing
    dense_parameters* params = (dense_parameters*)l->parameters;
    matrix_set(params->weights, 0, 0, 1.0);
    matrix_set(params->weights, 0, 1, 1.0);
    matrix_set(params->bias, 0, 0, 0.5);
    
    // Create input
    matrix* input = matrix_new(2, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);
    
    // Forward pass
    matrix* output = l->forward(l, input);
    
    // Expected output: (1.0 * 1.0 + 1.0 * 2.0 + 0.5) = 3.5
    cr_assert_not_null(output, "Forward pass returned NULL");
    cr_assert_float_eq(matrix_at(output, 0, 0), 3.5, 1e-6, "Forward pass calculation incorrect");
    
    matrix_free(input);
    matrix_free(output);
    layer_free(l);
}

Test(dense_layer_backward, backward_propagation) {
    unsigned int input_dim = 2;
    unsigned int output_dim = 1;
    layer* l = dense_layer_new(input_dim, output_dim, 0.0, 0.0);
    
    // Set weights and perform forward pass first
    dense_parameters* params = (dense_parameters*)l->parameters;
    matrix_set(params->weights, 0, 0, 1.0);
    matrix_set(params->weights, 0, 1, 1.0);
    matrix_set(params->bias, 0, 0, 0.5);
    
    matrix* input = matrix_new(2, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);
    
    matrix* output = l->forward(l, input);
    
    // Create gradient
    matrix* gradient = matrix_new(1, 1, sizeof(double));
    matrix_set(gradient, 0, 0, 1.0);
    
    // Backward pass
    matrix* input_gradient = l->backward(l, gradient);
    
    // Check input gradient
    cr_assert_not_null(input_gradient, "Backward pass returned NULL");
    cr_assert_float_eq(matrix_at(input_gradient, 0, 0), 1.0, 1e-6, "Input gradient[0] incorrect");
    cr_assert_float_eq(matrix_at(input_gradient, 1, 0), 1.0, 1e-6, "Input gradient[1] incorrect");
    
    // Check weight gradients
    cr_assert_float_eq(matrix_at(params->d_weights, 0, 0), 1.0, 1e-6, "Weight gradient[0] incorrect");
    cr_assert_float_eq(matrix_at(params->d_weights, 0, 1), 2.0, 1e-6, "Weight gradient[1] incorrect");
    
    matrix_free(input);
    matrix_free(output);
    matrix_free(gradient);
    matrix_free(input_gradient);
    layer_free(l);
}
