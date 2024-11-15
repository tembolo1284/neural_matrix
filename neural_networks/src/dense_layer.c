#include "dense_layer.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

// Forward declarations for wrapper functions
static matrix* dense_forward_wrapper(layer* l, ...);
static matrix* dense_backward_wrapper(layer* l, matrix* gradient);

layer* dense_layer_new(unsigned int input_dim, unsigned int output_dim, double weight_min, double weight_max) {
    fprintf(stderr, "Starting dense_layer_new...\n");

    if (input_dim == 0 || output_dim == 0) {
        fprintf(stderr, "Error: Invalid input_dim (%u) or output_dim (%u)\n", input_dim, output_dim);
        return NULL;
    }

    fprintf(stderr, "Allocating layer structure...\n");
    layer* l = layer_new(input_dim, output_dim);
    if (!l) {
        fprintf(stderr, "Error: Failed to allocate layer structure\n");
        return NULL;
    }
    fprintf(stderr, "Layer structure allocated successfully\n");

    fprintf(stderr, "Allocating dense parameters...\n");
    dense_parameters* params = calloc(1, sizeof(dense_parameters));
    if (!params) {
        fprintf(stderr, "Error: Failed to allocate dense parameters\n");
        layer_free(l);
        return NULL;
    }
    fprintf(stderr, "Dense parameters allocated successfully\n");

    fprintf(stderr, "Initializing weights...\n");
    params->weights = matrix_rand(output_dim, input_dim, weight_min, weight_max, sizeof(double));
    if (!params->weights) {
        fprintf(stderr, "Error: Failed to initialize weights matrix\n");
        free(params);
        layer_free(l);
        return NULL;
    }
    fprintf(stderr, "Weights initialized successfully\n");

    fprintf(stderr, "Initializing bias...\n");
    params->bias = matrix_new(output_dim, 1, sizeof(double));
    if (!params->bias) {
        fprintf(stderr, "Error: Failed to initialize bias matrix\n");
        matrix_free(params->weights);
        free(params);
        layer_free(l);
        return NULL;
    }
    fprintf(stderr, "Bias initialized successfully\n");

    params->input = NULL;
    params->output = NULL;
    params->d_weights = NULL;
    params->d_bias = NULL;

    fprintf(stderr, "Assigning parameters and function pointers to layer...\n");
    l->parameters = params;
    l->forward = dense_forward_wrapper;
    l->backward = dense_backward_wrapper;
    l->update = dense_update;
    l->free = dense_free;

    fprintf(stderr, "dense_layer_new completed successfully\n");
    return l;
}

// Wrapper function for forward propagation to match variadic signature
static matrix* dense_forward_wrapper(layer* l, ...) {
    va_list args;
    va_start(args, l);
    matrix* input = va_arg(args, matrix*);
    va_end(args);
    return dense_forward(l, input);
}

// Wrapper function for backward propagation (non-variadic, direct call)
static matrix* dense_backward_wrapper(layer* l, matrix* gradient) {
    return dense_backward(l, gradient);
}

matrix* dense_forward(layer* l, matrix* input) {
    fprintf(stderr, "Entering dense_forward...\n");

    if (!l || !input) {
        fprintf(stderr, "Error: Null layer or input in dense_forward\n");
        return NULL;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;

    fprintf(stderr, "Caching input matrix...\n");
    if (params->input) {
        matrix_free(params->input);
    }
    params->input = matrix_copy(input);
    if (!params->input) {
        fprintf(stderr, "Error: Failed to copy input matrix in dense_forward\n");
        return NULL;
    }

    fprintf(stderr, "Computing weighted sum (W * x)...\n");
    matrix* weighted = matrix_mult(params->weights, input);
    if (!weighted) {
        fprintf(stderr, "Error: Failed to multiply matrices in dense_forward\n");
        return NULL;
    }

    fprintf(stderr, "Adding bias to weighted sum...\n");
    for (unsigned int i = 0; i < weighted->num_rows; i++) {
        for (unsigned int j = 0; j < weighted->num_cols; j++) {
            double val = matrix_at(weighted, i, j) + matrix_at(params->bias, i, 0);
            matrix_set(weighted, i, j, val);
        }
    }

    fprintf(stderr, "Caching output matrix...\n");
    if (params->output) {
        matrix_free(params->output);
    }
    params->output = matrix_copy(weighted);
    if (!params->output) {
        matrix_free(weighted);
        fprintf(stderr, "Error: Failed to cache output in dense_forward\n");
        return NULL;
    }

    fprintf(stderr, "dense_forward completed successfully\n");
    return weighted;
}

matrix* dense_backward(layer* l, matrix* gradient) {
    fprintf(stderr, "Entering dense_backward...\n");

    if (!l || !gradient) {
        fprintf(stderr, "Error: Null layer or gradient in dense_backward\n");
        return NULL;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;
    if (!params->input) {
        fprintf(stderr, "Error: No input cached from forward pass in dense_backward\n");
        return NULL;
    }

    fprintf(stderr, "Computing input gradient (W^T * gradient)...\n");
    matrix* weights_T = matrix_copy(params->weights);
    if (!weights_T) {
        fprintf(stderr, "Error: Failed to copy weights matrix in dense_backward\n");
        return NULL;
    }
    matrix_transpose(weights_T);

    matrix* input_gradient = matrix_mult(weights_T, gradient);
    matrix_free(weights_T);
    if (!input_gradient) {
        fprintf(stderr, "Error: Failed to compute input gradient in dense_backward\n");
        return NULL;
    }

    fprintf(stderr, "Computing weight gradients (gradient * input^T)...\n");
    matrix* input_T = matrix_copy(params->input);
    if (!input_T) {
        matrix_free(input_gradient);
        fprintf(stderr, "Error: Failed to copy input matrix in dense_backward\n");
        return NULL;
    }
    matrix_transpose(input_T);

    if (params->d_weights) {
        matrix_free(params->d_weights);
    }
    params->d_weights = matrix_mult(gradient, input_T);
    matrix_free(input_T);
    if (!params->d_weights) {
        matrix_free(input_gradient);
        fprintf(stderr, "Error: Failed to compute weight gradients in dense_backward\n");
        return NULL;
    }

    fprintf(stderr, "Computing bias gradients...\n");
    if (params->d_bias) {
        matrix_free(params->d_bias);
    }
    params->d_bias = matrix_new(l->output_dim, 1, sizeof(double));
    if (!params->d_bias) {
        matrix_free(input_gradient);
        fprintf(stderr, "Error: Failed to create bias gradient matrix in dense_backward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < l->output_dim; i++) {
        double sum = 0.0;
        for (unsigned int j = 0; j < gradient->num_cols; j++) {
            sum += matrix_at(gradient, i, j);
        }
        matrix_set(params->d_bias, i, 0, sum);
    }

    fprintf(stderr, "dense_backward completed successfully\n");
    return input_gradient;
}

void dense_update(layer* l, double learning_rate) {
    fprintf(stderr, "Entering dense_update with learning rate: %f\n", learning_rate);

    if (!l || !l->parameters) {
        fprintf(stderr, "Error: Null layer or parameters in dense_update\n");
        return;
    }

    dense_parameters* params = (dense_parameters*)l->parameters;
    if (!params->d_weights || !params->d_bias) {
        fprintf(stderr, "Error: Null gradients in dense_update\n");
        return;
    }

    fprintf(stderr, "Updating weights and biases...\n");

    // Update weights
    matrix_mult_r(params->d_weights, -learning_rate);
    matrix* new_weights = matrix_add(params->weights, params->d_weights);
    matrix_free(params->weights);
    params->weights = new_weights;

    // Update bias
    matrix_mult_r(params->d_bias, -learning_rate);
    matrix* new_bias = matrix_add(params->bias, params->d_bias);
    matrix_free(params->bias);
    params->bias = new_bias;

    fprintf(stderr, "dense_update completed successfully\n");
}

void dense_free(layer* l) {
    fprintf(stderr, "Entering dense_free...\n");

    if (!l) {
        fprintf(stderr, "Warning: Attempted to free a NULL layer\n");
        return;
    }

    if (l->parameters) {
        fprintf(stderr, "Freeing dense layer parameters...\n");

        dense_parameters* params = (dense_parameters*)l->parameters;

        if (params->weights) {
            fprintf(stderr, "Freeing weights matrix...\n");
            matrix_free(params->weights);
            params->weights = NULL;
        }

        if (params->bias) {
            fprintf(stderr, "Freeing bias matrix...\n");
            matrix_free(params->bias);
            params->bias = NULL;
        }

        if (params->input) {
            fprintf(stderr, "Freeing input matrix...\n");
            matrix_free(params->input);
            params->input = NULL;
        }

        if (params->output) {
            fprintf(stderr, "Freeing output matrix...\n");
            matrix_free(params->output);
            params->output = NULL;
        }

        if (params->d_weights) {
            fprintf(stderr, "Freeing weight gradients...\n");
            matrix_free(params->d_weights);
            params->d_weights = NULL;
        }

        if (params->d_bias) {
            fprintf(stderr, "Freeing bias gradients...\n");
            matrix_free(params->d_bias);
            params->d_bias = NULL;
        }

        fprintf(stderr, "Freeing dense parameters structure...\n");
        free(params);
        l->parameters = NULL;
    }

    // Prevent recursion by clearing the function pointer
    fprintf(stderr, "Clearing l->free to prevent recursion...\n");
    l->free = NULL;

    fprintf(stderr, "Freeing layer structure...\n");
    layer_free(l);

    fprintf(stderr, "Exiting dense_free...\n");
}

