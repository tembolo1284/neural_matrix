#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include "../include/loss.h"
#include <math.h>

Test(loss_mse, forward_pass) {
    fprintf(stderr, "\nStarting MSE forward pass test...\n");
    unsigned int dim = 2;
    
    layer* l = loss_layer_new(dim, LOSS_MSE);
    cr_assert_not_null(l, "Failed to create loss layer");
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    cr_assert_not_null(predicted, "Failed to create predicted matrix");
    cr_assert_not_null(target, "Failed to create target matrix");
    
    matrix_set(predicted, 0, 0, 1.0);
    matrix_set(predicted, 1, 0, 2.0);
    matrix_set(target, 0, 0, 2.0);
    matrix_set(target, 1, 0, 1.0);
    
    matrix* loss = l->forward(l, predicted, target);
    cr_assert_not_null(loss, "Forward pass returned NULL");
    
    // MSE should be ((1-2)^2 + (2-1)^2) / 2 = 1
    cr_assert(fabs(matrix_at(loss, 0, 0) - 1.0) < 1e-6, "MSE calculation incorrect");
    
    fprintf(stderr, "Cleaning up MSE forward test resources...\n");
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    l->free(l);  // Use the layer's free function directly instead of loss_free
    fprintf(stderr, "MSE forward test cleanup complete\n");
}

Test(loss_mse, backward_pass) {
    fprintf(stderr, "\nStarting MSE backward pass test...\n");
    unsigned int dim = 2;
    
    layer* l = loss_layer_new(dim, LOSS_MSE);
    cr_assert_not_null(l, "Failed to create loss layer");
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    cr_assert_not_null(predicted, "Failed to create predicted matrix");
    cr_assert_not_null(target, "Failed to create target matrix");
    
    matrix_set(predicted, 0, 0, 1.0);
    matrix_set(predicted, 1, 0, 2.0);
    matrix_set(target, 0, 0, 2.0);
    matrix_set(target, 1, 0, 1.0);
    
    matrix* loss = l->forward(l, predicted, target);
    cr_assert_not_null(loss, "Forward pass returned NULL");
    
    matrix* gradient = l->backward(l, NULL);
    cr_assert_not_null(gradient, "Backward pass returned NULL");
    
    cr_assert(fabs(matrix_at(gradient, 0, 0) + 0.5) < 1e-6, "MSE gradient[0] incorrect");
    cr_assert(fabs(matrix_at(gradient, 1, 0) - 0.5) < 1e-6, "MSE gradient[1] incorrect");
    
    fprintf(stderr, "Cleaning up MSE backward test resources...\n");
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    matrix_free(gradient);
    l->free(l);  // Use the layer's free function directly
    fprintf(stderr, "MSE backward test cleanup complete\n");
}

Test(loss_bce, forward_pass) {
    fprintf(stderr, "\nStarting BCE forward pass test...\n");
    unsigned int dim = 2;
    
    layer* l = loss_layer_new(dim, LOSS_BINARY_CROSS_ENTROPY);
    cr_assert_not_null(l, "Failed to create loss layer");
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    cr_assert_not_null(predicted, "Failed to create predicted matrix");
    cr_assert_not_null(target, "Failed to create target matrix");
    
    matrix_set(predicted, 0, 0, 0.8);  // Prediction close to 1
    matrix_set(predicted, 1, 0, 0.1);  // Prediction close to 0
    matrix_set(target, 0, 0, 1.0);     // True positive
    matrix_set(target, 1, 0, 0.0);     // True negative
    
    matrix* loss = l->forward(l, predicted, target);
    cr_assert_not_null(loss, "Forward pass returned NULL");
    
    cr_assert(matrix_at(loss, 0, 0) > 0.0, "BCE loss should be positive");
    
    fprintf(stderr, "Cleaning up BCE test resources...\n");
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    l->free(l);  // Use the layer's free function directly
    fprintf(stderr, "BCE test cleanup complete\n");
}

Test(loss_ce, forward_pass) {
    fprintf(stderr, "\nStarting CE forward pass test...\n");
    unsigned int dim = 3;
    
    layer* l = loss_layer_new(dim, LOSS_CROSS_ENTROPY);
    cr_assert_not_null(l, "Failed to create loss layer");
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    cr_assert_not_null(predicted, "Failed to create predicted matrix");
    cr_assert_not_null(target, "Failed to create target matrix");
    
    // Softmax-like predictions
    matrix_set(predicted, 0, 0, 0.7);
    matrix_set(predicted, 1, 0, 0.2);
    matrix_set(predicted, 2, 0, 0.1);
    
    // One-hot encoded target
    matrix_set(target, 0, 0, 1.0);
    matrix_set(target, 1, 0, 0.0);
    matrix_set(target, 2, 0, 0.0);
    
    matrix* loss = l->forward(l, predicted, target);
    cr_assert_not_null(loss, "Forward pass returned NULL");
    
    cr_assert(matrix_at(loss, 0, 0) > 0.0, "Cross entropy loss should be positive");
    
    fprintf(stderr, "Cleaning up CE test resources...\n");
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    l->free(l);  // Use the layer's free function directly
    fprintf(stderr, "CE test cleanup complete\n");
}
