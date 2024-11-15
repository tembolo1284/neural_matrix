#include <criterion/criterion.h>
#include <criterion/new/assert.h>
#include "../include/optimization.h"
#include "../include/dense_layer.h"

// Helper function to create and prepare a test layer
static layer* create_test_layer() {
    fprintf(stderr, "Creating test layer...\n");
    layer* l = dense_layer_new(2, 1, -1.0, 1.0);
    if (!l) {
        fprintf(stderr, "Failed to create test layer\n");
        return NULL;
    }
    
    // Initialize gradients
    dense_parameters* params = (dense_parameters*)l->parameters;
    fprintf(stderr, "Creating gradient matrices...\n");
    
    // Create d_weights if it doesn't exist
    if (!params->d_weights) {
        params->d_weights = matrix_new(1, 2, sizeof(double));
        if (!params->d_weights) {
            fprintf(stderr, "Failed to create d_weights\n");
            dense_free(l);
            return NULL;
        }
    }
    
    // Create d_bias if it doesn't exist
    if (!params->d_bias) {
        params->d_bias = matrix_new(1, 1, sizeof(double));
        if (!params->d_bias) {
            fprintf(stderr, "Failed to create d_bias\n");
            dense_free(l);
            return NULL;
        }
    }

    // Initialize input cache to prevent null pointer issues
    params->input = matrix_new(2, 1, sizeof(double));
    if (!params->input) {
        fprintf(stderr, "Failed to create input cache\n");
        dense_free(l);
        return NULL;
    }

    fprintf(stderr, "Test layer created and initialized successfully\n");
    return l;
}

// Helper function to set gradients
static void set_gradients(dense_parameters* params) {
    fprintf(stderr, "Setting gradients...\n");
    matrix_set(params->d_weights, 0, 0, 1.0);
    matrix_set(params->d_weights, 0, 1, 1.0);
    matrix_set(params->d_bias, 0, 0, 1.0);
    fprintf(stderr, "Gradients set successfully\n");
}

Test(optimizer, sgd_basic_update) {
    fprintf(stderr, "\nStarting SGD basic update test...\n");
    
    // Create and prepare test layer
    layer* l = create_test_layer();
    cr_assert_not_null(l, "Failed to create layer");
    
    dense_parameters* params = (dense_parameters*)l->parameters;
    cr_assert_not_null(params, "Layer parameters are null");
    cr_assert_not_null(params->weights, "Weights are null");
    cr_assert_not_null(params->d_weights, "Gradient weights are null");
    
    // Store initial weights
    double w1 = matrix_at(params->weights, 0, 0);
    double w2 = matrix_at(params->weights, 0, 1);
    fprintf(stderr, "Initial weights: w1=%f, w2=%f\n", w1, w2);
    
    // Set gradients
    set_gradients(params);
    
    // Create optimizer with single layer
    layer** layers = malloc(sizeof(layer*));
    layers[0] = l;
    
    optimizer* opt = optimizer_new(OPTIMIZER_SGD, 0.1, 1, layers);
    cr_assert_not_null(opt, "Failed to create optimizer");
    
    // Perform update
    fprintf(stderr, "Performing optimizer step...\n");
    optimizer_step(opt);
    
    // Check weight updates
    double new_w1 = matrix_at(params->weights, 0, 0);
    double new_w2 = matrix_at(params->weights, 0, 1);
    fprintf(stderr, "New weights: w1=%f, w2=%f\n", new_w1, new_w2);
    
    cr_assert_float_eq(new_w1, w1 - 0.1, 1e-6, "Weight update incorrect");
    cr_assert_float_eq(new_w2, w2 - 0.1, 1e-6, "Weight update incorrect");
    
    // Clean up
    fprintf(stderr, "Cleaning up SGD test resources...\n");
    // First free the optimizer (but don't let it free our layer)
    free(opt->layers); // Free the layers array but not the layer itself
    opt->layers = NULL; // Prevent optimizer from trying to access layers during free
    optimizer_free(opt);
    
    // Then free the layer
    fprintf(stderr, "Freeing layer...\n");
    dense_free(l);
    fprintf(stderr, "SGD test cleanup complete\n");
}

Test(optimizer, momentum_basic_update) {
    fprintf(stderr, "\nStarting Momentum basic update test...\n");
    
    // Create and prepare test layer
    layer* l = create_test_layer();
    cr_assert_not_null(l, "Failed to create layer");
    
    dense_parameters* params = (dense_parameters*)l->parameters;
    cr_assert_not_null(params, "Layer parameters are null");
    
    // Store initial weights
    double w1 = matrix_at(params->weights, 0, 0);
    double w2 = matrix_at(params->weights, 0, 1);
    fprintf(stderr, "Initial weights: w1=%f, w2=%f\n", w1, w2);
    
    // Set gradients
    set_gradients(params);
    
    // Create optimizer
    layer** layers = malloc(sizeof(layer*));
    layers[0] = l;
    
    optimizer* opt = optimizer_new(OPTIMIZER_MOMENTUM, 0.1, 1, layers);
    cr_assert_not_null(opt, "Failed to create optimizer");
    opt->momentum = 0.9;
    
    // Perform update
    fprintf(stderr, "Performing optimizer step...\n");
    optimizer_step(opt);
    
    // Clean up
    fprintf(stderr, "Cleaning up Momentum test resources...\n");
    free(opt->layers);
    opt->layers = NULL;
    optimizer_free(opt);
    dense_free(l);
    fprintf(stderr, "Momentum test cleanup complete\n");
}

Test(optimizer, rmsprop_basic_update) {
    fprintf(stderr, "\nStarting RMSprop basic update test...\n");
    
    // Create and prepare test layer
    layer* l = create_test_layer();
    cr_assert_not_null(l, "Failed to create layer");
    
    dense_parameters* params = (dense_parameters*)l->parameters;
    cr_assert_not_null(params, "Layer parameters are null");
    
    // Set gradients
    set_gradients(params);
    
    // Create optimizer
    layer** layers = malloc(sizeof(layer*));
    layers[0] = l;
    
    optimizer* opt = optimizer_new(OPTIMIZER_RMSPROP, 0.1, 1, layers);
    cr_assert_not_null(opt, "Failed to create optimizer");
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    
    // Perform update
    fprintf(stderr, "Performing optimizer step...\n");
    optimizer_step(opt);
    
    // Clean up
    fprintf(stderr, "Cleaning up RMSprop test resources...\n");
    free(opt->layers);
    opt->layers = NULL;
    optimizer_free(opt);
    dense_free(l);
    fprintf(stderr, "RMSprop test cleanup complete\n");
}

Test(optimizer, adam_basic_update) {
    fprintf(stderr, "\nStarting Adam basic update test...\n");
    
    // Create and prepare test layer
    layer* l = create_test_layer();
    cr_assert_not_null(l, "Failed to create layer");
    
    dense_parameters* params = (dense_parameters*)l->parameters;
    cr_assert_not_null(params, "Layer parameters are null");
    
    // Set gradients
    set_gradients(params);
    
    // Create optimizer
    layer** layers = malloc(sizeof(layer*));
    layers[0] = l;
    
    optimizer* opt = optimizer_new(OPTIMIZER_ADAM, 0.1, 1, layers);
    cr_assert_not_null(opt, "Failed to create optimizer");
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    
    // Perform update
    fprintf(stderr, "Performing optimizer step...\n");
    optimizer_step(opt);
    
    // Clean up
    fprintf(stderr, "Cleaning up Adam test resources...\n");
    free(opt->layers);
    opt->layers = NULL;
    optimizer_free(opt);
    dense_free(l);
    fprintf(stderr, "Adam test cleanup complete\n");
}
