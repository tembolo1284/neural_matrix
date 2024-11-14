#include <criterion/criterion.h>
#include <criterion/logging.h>
#include "../include/optimization.h"
#include "../include/dense_layer.h"

// Helper function to create a simple network layer
static layer* create_test_layer(void) {
    return dense_layer_new(2, 1, -1.0, 1.0);
}

Test(optimizer_init, creation_and_setup) {
    optimizer* opt = optimizer_new(OPTIMIZER_SGD, 0.01);
    
    cr_assert_not_null(opt, "Optimizer creation failed");
    cr_assert_eq(opt->type, OPTIMIZER_SGD, "Optimizer type mismatch");
    cr_assert_float_eq(opt->learning_rate, 0.01, 1e-6, "Learning rate mismatch");
    cr_assert_eq(opt->num_layers, 0, "Initial layer count should be 0");
    
    // Test adding a layer
    layer* l = create_test_layer();
    optimizer_add_layer(opt, l);
    cr_assert_eq(opt->num_layers, 1, "Layer count mismatch after adding layer");
    cr_assert_eq(opt->layers[0], l, "Layer pointer mismatch");
    
    optimizer_free(opt);
    layer_free(l);
}

Test(optimizer_sgd, basic_update) {
    optimizer* opt = optimizer_new(OPTIMIZER_SGD, 0.1);
    layer* l = create_test_layer();
    optimizer_add_layer(opt, l);
    optimizer_init(opt);
    
    // Create a simple gradient
    dense_parameters* params = (dense_parameters*)l->parameters;
    params->d_weights = matrix_new(1, 2, sizeof(double));
    matrix_set(params->d_weights, 0, 0, 1.0);
    matrix_set(params->d_weights, 0, 1, 1.0);
    
    // Store initial weights
    double w1 = matrix_at(params->weights, 0, 0);
    double w2 = matrix_at(params->weights, 0, 1);
    
    // Perform update
    optimizer_step(opt);
    
    // Check that weights were updated correctly
    cr_assert_float_eq(matrix_at(params->weights, 0, 0), w1 - 0.1, 1e-6, "Weight update incorrect");
    cr_assert_float_eq(matrix_at(params->weights, 0, 1), w2 - 0.1, 1e-6, "Weight update incorrect");
    
    optimizer_free(opt);
    layer_free(l);
}

Test(optimizer_adam, initialization) {
    optimizer* opt = optimizer_new(OPTIMIZER_ADAM, 0.001);
    layer* l = create_test_layer();
    
    cr_assert_not_null(opt, "Adam optimizer creation failed");
    cr_assert_eq(opt->type, OPTIMIZER_ADAM, "Optimizer type mismatch");
    cr_assert_float_eq(opt->learning_rate, 0.001, 1e-6, "Learning rate mismatch");
    cr_assert_float_eq(opt->beta1, 0.9, 1e-6, "Beta1 default value mismatch");
    cr_assert_float_eq(opt->beta2, 0.999, 1e-6, "Beta2 default value mismatch");
    
    optimizer_add_layer(opt, l);
    optimizer_init(opt);
    
    cr_assert_not_null(opt->moment, "Moment array should be initialized");
    cr_assert_not_null(opt->cache, "Cache array should be initialized");
    
    optimizer_free(opt);
    layer_free(l);
}
