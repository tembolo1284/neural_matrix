#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/loss.h"
#include <math.h>

// MSE Loss Tests
Test(loss_mse, forward_pass) {
    unsigned int dim = 2;
    layer* l = loss_layer_new(dim, LOSS_MSE);
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    
    matrix_set(predicted, 0, 0, 1.0);
    matrix_set(predicted, 1, 0, 2.0);
    matrix_set(target, 0, 0, 2.0);
    matrix_set(target, 1, 0, 1.0);
    
    matrix* loss = l->forward(l, predicted, target);
    
    // MSE should be ((1-2)^2 + (2-1)^2) / 2 = 1
    cr_assert_float_eq(matrix_at(loss, 0, 0), 1.0, 1e-6, "MSE calculation incorrect");
    
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    layer_free(l);
}

Test(loss_mse, backward_pass) {
    unsigned int dim = 2;
    layer* l = loss_layer_new(dim, LOSS_MSE);
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    
    matrix_set(predicted, 0, 0, 1.0);
    matrix_set(predicted, 1, 0, 2.0);
    matrix_set(target, 0, 0, 2.0);
    matrix_set(target, 1, 0, 1.0);
    
    // Forward pass to set up internal state
    matrix* loss = l->forward(l, predicted, target);
    matrix* gradient = l->backward(l);
    
    // Gradients should be (pred - target) / dim
    cr_assert_float_eq(matrix_at(gradient, 0, 0), -0.5, 1e-6, "MSE gradient[0] incorrect");
    cr_assert_float_eq(matrix_at(gradient, 1, 0), 0.5, 1e-6, "MSE gradient[1] incorrect");
    
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    matrix_free(gradient);
    layer_free(l);
}

// Binary Cross Entropy Tests
Test(loss_bce, forward_pass) {
    unsigned int dim = 2;
    layer* l = loss_layer_new(dim, LOSS_BINARY_CROSS_ENTROPY);
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    
    matrix_set(predicted, 0, 0, 0.8);  // Prediction close to 1
    matrix_set(predicted, 1, 0, 0.1);  // Prediction close to 0
    matrix_set(target, 0, 0, 1.0);     // True positive
    matrix_set(target, 1, 0, 0.0);     // True negative
    
    matrix* loss = l->forward(l, predicted, target);
    
    cr_assert_float_gt(matrix_at(loss, 0, 0), 0.0, "BCE loss should be positive");
    
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    layer_free(l);
}

// Cross Entropy Tests
Test(loss_ce, forward_pass) {
    unsigned int dim = 3;
    layer* l = loss_layer_new(dim, LOSS_CROSS_ENTROPY);
    
    matrix* predicted = matrix_new(dim, 1, sizeof(double));
    matrix* target = matrix_new(dim, 1, sizeof(double));
    
    // Softmax-like predictions
    matrix_set(predicted, 0, 0, 0.7);
    matrix_set(predicted, 1, 0, 0.2);
    matrix_set(predicted, 2, 0, 0.1);
    
    // One-hot encoded target
    matrix_set(target, 0, 0, 1.0);
    matrix_set(target, 1, 0, 0.0);
    matrix_set(target, 2, 0, 0.0);
    
    matrix* loss = l->forward(l, predicted, target);
    
    cr_assert_float_gt(matrix_at(loss, 0, 0), 0.0, "Cross entropy loss should be positive");
    
    matrix_free(predicted);
    matrix_free(target);
    matrix_free(loss);
    layer_free(l);
}
