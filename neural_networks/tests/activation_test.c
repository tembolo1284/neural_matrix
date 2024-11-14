#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/activation.h"
#include <math.h>

// ReLU Tests
Test(activation_relu, forward_pass) {
    unsigned int dim = 3;
    layer* l = activation_layer_new(dim, ACTIVATION_RELU);
    
    matrix* input = matrix_new(dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);   // Positive
    matrix_set(input, 1, 0, -2.0);  // Negative
    matrix_set(input, 2, 0, 0.0);   // Zero
    
    matrix* output = l->forward(l, input);
    
    cr_assert_float_eq(matrix_at(output, 0, 0), 1.0, 1e-6, "ReLU failed for positive input");
    cr_assert_float_eq(matrix_at(output, 1, 0), 0.0, 1e-6, "ReLU failed for negative input");
    cr_assert_float_eq(matrix_at(output, 2, 0), 0.0, 1e-6, "ReLU failed for zero input");
    
    matrix_free(input);
    matrix_free(output);
    layer_free(l);
}

Test(activation_relu, backward_pass) {
    unsigned int dim = 3;
    layer* l = activation_layer_new(dim, ACTIVATION_RELU);
    
    matrix* input = matrix_new(dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);   // Positive
    matrix_set(input, 1, 0, -2.0);  // Negative
    matrix_set(input, 2, 0, 0.0);   // Zero
    
    // Forward pass to set up internal state
    matrix* output = l->forward(l, input);
    
    // Create gradient
    matrix* gradient = matrix_new(dim, 1, sizeof(double));
    matrix_set(gradient, 0, 0, 1.0);
    matrix_set(gradient, 1, 0, 1.0);
    matrix_set(gradient, 2, 0, 1.0);
    
    matrix* backward = l->backward(l, gradient);
    
    cr_assert_float_eq(matrix_at(backward, 0, 0), 1.0, 1e-6, "ReLU gradient failed for positive input");
    cr_assert_float_eq(matrix_at(backward, 1, 0), 0.0, 1e-6, "ReLU gradient failed for negative input");
    cr_assert_float_eq(matrix_at(backward, 2, 0), 0.0, 1e-6, "ReLU gradient failed for zero input");
    
    matrix_free(input);
    matrix_free(output);
    matrix_free(gradient);
    matrix_free(backward);
    layer_free(l);
}

// Sigmoid Tests
Test(activation_sigmoid, forward_pass) {
    unsigned int dim = 3;
    layer* l = activation_layer_new(dim, ACTIVATION_SIGMOID);
    
    matrix* input = matrix_new(dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 0.0);   // Should give 0.5
    matrix_set(input, 1, 0, 4.0);   // Should be close to 1
    matrix_set(input, 2, 0, -4.0);  // Should be close to 0
    
    matrix* output = l->forward(l, input);
    
    cr_assert_float_eq(matrix_at(output, 0, 0), 0.5, 1e-6, "Sigmoid failed for zero input");
    cr_assert_float_eq(matrix_at(output, 1, 0), 0.982013790037908, 1e-6, "Sigmoid failed for large positive input");
    cr_assert_float_eq(matrix_at(output, 2, 0), 0.017986209962091559, 1e-6, "Sigmoid failed for large negative input");
    
    matrix_free(input);
    matrix_free(output);
    layer_free(l);
}

// Tanh Tests
Test(activation_tanh, forward_pass) {
    unsigned int dim = 3;
    layer* l = activation_layer_new(dim, ACTIVATION_TANH);
    
    matrix* input = matrix_new(dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 0.0);   // Should give 0
    matrix_set(input, 1, 0, 2.0);   // Should be close to 1
    matrix_set(input, 2, 0, -2.0);  // Should be close to -1
    
    matrix* output = l->forward(l, input);
    
    cr_assert_float_eq(matrix_at(output, 0, 0), 0.0, 1e-6, "Tanh failed for zero input");
    cr_assert_float_eq(matrix_at(output, 1, 0), tanh(2.0), 1e-6, "Tanh failed for positive input");
    cr_assert_float_eq(matrix_at(output, 2, 0), tanh(-2.0), 1e-6, "Tanh failed for negative input");
    
    matrix_free(input);
    matrix_free(output);
    layer_free(l);
}

// Softmax Tests
Test(activation_softmax, forward_pass) {
    unsigned int dim = 3;
    layer* l = activation_layer_new(dim, ACTIVATION_SOFTMAX);
    
    matrix* input = matrix_new(dim, 1, sizeof(double));
    matrix_set(input, 0, 0, 1.0);
    matrix_set(input, 1, 0, 2.0);
    matrix_set(input, 2, 0, 3.0);
    
    matrix* output = l->forward(l, input);
    
    // Sum should be 1
    double sum = matrix_at(output, 0, 0) + matrix_at(output, 1, 0) + matrix_at(output, 2, 0);
    cr_assert_float_eq(sum, 1.0, 1e-6, "Softmax outputs should sum to 1");
    
    // Each output should be positive
    cr_assert_float_gt(matrix_at(output, 0, 0), 0.0, "Softmax output should be positive");
    cr_assert_float_gt(matrix_at(output, 1, 0), 0.0, "Softmax output should be positive");
    cr_assert_float_gt(matrix_at(output, 2, 0), 0.0, "Softmax output should be positive");
    
    // Ordering should be preserved
    cr_assert_float_lt(matrix_at(output, 0, 0), matrix_at(output, 1, 0), "Softmax should preserve ordering");
    cr_assert_float_lt(matrix_at(output, 1, 0), matrix_at(output, 2, 0), "Softmax should preserve ordering");
    
    matrix_free(input);
    matrix_free(output);
    layer_free(l);
}
