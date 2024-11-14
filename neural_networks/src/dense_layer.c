#include "dense_layer.h"
#include <stdlib.h>

layer* dense_layer_new(unsigned int input_dim, 
                      unsigned int output_dim,
                      double weight_init_min,
                      double weight_init_max) {
    
    log_debug("Creating dense layer with input_dim=%u, output_dim=%u", 
              input_dim, output_dim);

    // Create base layer
    layer* l = layer_new(input_dim, output_dim);
    if (!l) {
        log_error("Failed to create base layer for dense layer");
        return NULL;
    }

    // Allocate dense parameters
    dense_parameters* params = calloc(1, sizeof(dense_parameters));
    if (!params) {
        log_error("Failed to allocate dense parameters");
        layer_free(l);
        return NULL;
    }

    // Initialize weights randomly
    params->weights = matrix_rand(output_dim, input_dim, 
                                weight_init_min, weight_init_max, 
                                sizeof(double));
    if (!params->weights) {
        log_error("Failed to create weight matrix");
        free(params);
        layer_free(l);
        return NULL;
    }

    // Initialize bias to zeros
    params->bias = matrix_new(output_dim, 1, sizeof(double));
    if (!params->bias) {
        log_error("Failed to create bias vector");
        matrix_free(params->weights);
        free(params);
        layer_free(l);
        return NULL;
    }

    // Initialize other matrices to NULL (will be created during forward/backward passes)
    params->input = NULL;
    params->output = NULL;
    params->d_weights = NULL;
    params->d_bias = NULL;

    // Set layer parameters and functions
    l->parameters = params;
    l->forward = dense_forward;
    l->backward = dense_backward;
    l->update = dense_update;
    l->free = dense_free;

    log_info("Successfully created dense layer");
    return l;
}

matrix* dense_forward(layer* l, matrix* input) {
    if (!l || !input) {
        log_error("NULL layer or input in dense_forward");
        return NULL;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;
    
    // Cache input for backprop
    if (params->input) {
        matrix_free(params->input);
    }
    params->input = matrix_copy(input);
    
    // Compute output = weights * input + bias
    matrix* weighted = matrix_mult(params->weights, input);
    if (!weighted) {
        log_error("Failed to compute weights * input");
        return NULL;
    }

    // Add bias to each column
    for (unsigned int i = 0; i < weighted->num_rows; i++) {
        for (unsigned int j = 0; j < weighted->num_cols; j++) {
            double val = matrix_at(weighted, i, j) + matrix_at(params->bias, i, 0);
            matrix_set(weighted, i, j, val);
        }
    }

    // Cache output for backprop
    if (params->output) {
        matrix_free(params->output);
    }
    params->output = matrix_copy(weighted);

    log_debug("Completed dense layer forward pass");
    return weighted;
}

matrix* dense_backward(layer* l, matrix* gradient) {
    if (!l || !gradient) {
        log_error("NULL layer or gradient in dense_backward");
        return NULL;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;
    
    // Compute weight gradients
    matrix* input_T = matrix_copy(params->input);
    matrix_transpose(input_T);
    
    if (params->d_weights) {
        matrix_free(params->d_weights);
    }
    params->d_weights = matrix_mult(gradient, input_T);
    matrix_free(input_T);

    // Compute bias gradients
    if (params->d_bias) {
        matrix_free(params->d_bias);
    }
    params->d_bias = matrix_new(l->output_dim, 1, sizeof(double));
    
    // Sum gradients across batch for bias
    for (unsigned int i = 0; i < gradient->num_rows; i++) {
        double sum = 0;
        for (unsigned int j = 0; j < gradient->num_cols; j++) {
            sum += matrix_at(gradient, i, j);
        }
        matrix_set(params->d_bias, i, 0, sum);
    }

    // Compute input gradients
    matrix* weights_T = matrix_copy(params->weights);
    matrix_transpose(weights_T);
    matrix* input_gradient = matrix_mult(weights_T, gradient);
    matrix_free(weights_T);

    log_debug("Completed dense layer backward pass");
    return input_gradient;
}

void dense_update(layer* l, double learning_rate) {
    if (!l) {
        log_error("NULL layer in dense_update");
        return;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;

    // Update weights: weights -= learning_rate * d_weights
    matrix_mult_r(params->d_weights, -learning_rate);
    matrix* new_weights = matrix_add(params->weights, params->d_weights);
    matrix_free(params->weights);
    params->weights = new_weights;

    // Update bias: bias -= learning_rate * d_bias
    matrix_mult_r(params->d_bias, -learning_rate);
    matrix* new_bias = matrix_add(params->bias, params->d_bias);
    matrix_free(params->bias);
    params->bias = new_bias;

    log_debug("Updated dense layer parameters with learning_rate=%f", learning_rate);
}

void dense_free(layer* l) {
    if (!l) {
        log_warn("Attempted to free NULL dense layer");
        return;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;
    if (params) {
        if (params->weights) matrix_free(params->weights);
        if (params->bias) matrix_free(params->bias);
        if (params->input) matrix_free(params->input);
        if (params->output) matrix_free(params->output);
        if (params->d_weights) matrix_free(params->d_weights);
        if (params->d_bias) matrix_free(params->d_bias);
    }

    log_debug("Freed dense layer resources");
}
