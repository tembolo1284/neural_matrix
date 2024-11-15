#include "loss.h"
#include <stdarg.h>
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

// Forward declarations of internal functions
static matrix* loss_forward(layer* l, matrix* predicted, matrix* target);
static matrix* loss_backward(layer* l);

// Wrapper function for the layer interface to handle variadic arguments
static matrix* loss_forward_wrapper(layer* l, ...) {
    fprintf(stderr, "Entering loss_forward_wrapper...\n");
    va_list args;
    va_start(args, l);
    matrix* predicted = va_arg(args, matrix*);
    matrix* target = va_arg(args, matrix*);
    va_end(args);
    fprintf(stderr, "Calling loss_forward with predicted and target matrices\n");
    return loss_forward(l, predicted, target);
}

static matrix* loss_backward_wrapper(layer* l, matrix* unused) {
    fprintf(stderr, "Entering loss_backward_wrapper...\n");
    (void)unused; // Suppress unused parameter warning
    return loss_backward(l);
}

layer* loss_layer_new(unsigned int dim, loss_type type) {
    fprintf(stderr, "Creating new loss layer with dim=%u and type=%d...\n", dim, type);
    
    layer* l = layer_new(dim, 1); // Loss outputs a scalar
    if (!l) {
        fprintf(stderr, "Failed to create base layer\n");
        return NULL;
    }

    loss_parameters* params = malloc(sizeof(loss_parameters));
    if (!params) {
        fprintf(stderr, "Failed to allocate loss parameters\n");
        layer_free(l);
        return NULL;
    }

    params->type = type;
    params->predicted = NULL;
    params->target = NULL;
    
    l->parameters = params;
    l->forward = loss_forward_wrapper;
    l->backward = loss_backward_wrapper;
    l->free = loss_free;
    
    fprintf(stderr, "Loss layer created successfully\n");
    return l;
}

// Helper functions
static double mse_loss_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating MSE loss...\n");
    double sum = 0.0;
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double p = matrix_at(predicted, i, 0);
        double t = matrix_at(target, i, 0);
        double diff = p - t;
        sum += diff * diff;
        fprintf(stderr, "Row %u: pred=%f, target=%f, diff=%f, diff²=%f\n", 
                i, p, t, diff, diff * diff);
    }
    double loss = sum / predicted->num_rows;
    fprintf(stderr, "Final MSE loss: %f (sum=%f, n=%u)\n", 
            loss, sum, predicted->num_rows);
    return loss;
}

static matrix* mse_gradient_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating MSE gradients...\n");
    matrix* gradient = matrix_new(predicted->num_rows, 1, sizeof(double));
    if (!gradient) {
        fprintf(stderr, "Failed to allocate gradient matrix\n");
        return NULL;
    }

    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double diff = matrix_at(predicted, i, 0) - matrix_at(target, i, 0);
        double grad = diff / predicted->num_rows;
        matrix_set(gradient, i, 0, grad);
        fprintf(stderr, "Row %u: gradient=%f\n", i, grad);
    }
    return gradient;
}

static double binary_cross_entropy_loss_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating BCE loss...\n");
    double sum = 0.0;
    const double epsilon = 1e-15; // Small constant to prevent log(0)
    
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double p = matrix_at(predicted, i, 0);
        double t = matrix_at(target, i, 0);
        
        // Clip predictions to prevent log(0)
        p = fmax(fmin(p, 1.0 - epsilon), epsilon);
        
        sum += t * log(p) + (1 - t) * log(1 - p);
        fprintf(stderr, "Row %u: pred=%f, target=%f, partial_sum=%f\n", 
                i, p, t, sum);
    }
    
    double loss = -sum / predicted->num_rows;
    fprintf(stderr, "Final BCE loss: %f\n", loss);
    return loss;
}

static matrix* binary_cross_entropy_gradient_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating BCE gradients...\n");
    matrix* gradient = matrix_new(predicted->num_rows, 1, sizeof(double));
    if (!gradient) {
        fprintf(stderr, "Failed to allocate gradient matrix\n");
        return NULL;
    }

    const double epsilon = 1e-15;
    
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double p = matrix_at(predicted, i, 0);
        double t = matrix_at(target, i, 0);
        
        // Clip predictions to prevent division by zero
        p = fmax(fmin(p, 1.0 - epsilon), epsilon);
        
        double grad = -(t/p - (1-t)/(1-p)) / predicted->num_rows;
        matrix_set(gradient, i, 0, grad);
        fprintf(stderr, "Row %u: gradient=%f\n", i, grad);
    }
    
    return gradient;
}

static double cross_entropy_loss_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating CE loss...\n");
    double sum = 0.0;
    const double epsilon = 1e-15;
    
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double p = matrix_at(predicted, i, 0);
        double t = matrix_at(target, i, 0);
        
        p = fmax(fmin(p, 1.0 - epsilon), epsilon);
        if (t > 0) { // Only calculate for positive targets (one-hot encoding)
            sum += t * log(p);
            fprintf(stderr, "Row %u: pred=%f, target=%f, partial_sum=%f\n", 
                    i, p, t, sum);
        }
    }
    
    double loss = -sum;
    fprintf(stderr, "Final CE loss: %f\n", loss);
    return loss;
}

static matrix* cross_entropy_gradient_calc(matrix* predicted, matrix* target) {
    fprintf(stderr, "Calculating CE gradients...\n");
    matrix* gradient = matrix_new(predicted->num_rows, 1, sizeof(double));
    if (!gradient) {
        fprintf(stderr, "Failed to allocate gradient matrix\n");
        return NULL;
    }

    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        double p = matrix_at(predicted, i, 0);
        double t = matrix_at(target, i, 0);
        double grad = -t/p;
        matrix_set(gradient, i, 0, grad);
        fprintf(stderr, "Row %u: gradient=%f\n", i, grad);
    }
    
    return gradient;
}

static matrix* loss_forward(layer* l, matrix* predicted, matrix* target) {
    fprintf(stderr, "Entering loss_forward...\n");
    if (!l || !predicted || !target) {
        fprintf(stderr, "Null pointer passed to loss_forward\n");
        return NULL;
    }

    loss_parameters* params = (loss_parameters*)l->parameters;
    
    // Clean up previous cached matrices
    if (params->predicted) {
        fprintf(stderr, "Freeing previous predicted matrix\n");
        matrix_free(params->predicted);
    }
    if (params->target) {
        fprintf(stderr, "Freeing previous target matrix\n");
        matrix_free(params->target);
    }
    
    // Cache inputs for backward pass
    fprintf(stderr, "Caching predicted and target matrices\n");
    params->predicted = matrix_copy(predicted);
    params->target = matrix_copy(target);
    
    if (!params->predicted || !params->target) {
        fprintf(stderr, "Failed to cache matrices\n");
        if (params->predicted) matrix_free(params->predicted);
        if (params->target) matrix_free(params->target);
        return NULL;
    }
    
    // Create output matrix for the loss value
    matrix* loss = matrix_new(1, 1, sizeof(double));
    if (!loss) {
        fprintf(stderr, "Failed to create loss matrix\n");
        return NULL;
    }
    
    double loss_value = 0.0;
    fprintf(stderr, "Calculating loss for type %d\n", params->type);
    switch (params->type) {
        case LOSS_MSE:
            loss_value = mse_loss_calc(predicted, target);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            loss_value = binary_cross_entropy_loss_calc(predicted, target);
            break;
        case LOSS_CROSS_ENTROPY:
            loss_value = cross_entropy_loss_calc(predicted, target);
            break;
        default:
            fprintf(stderr, "Unsupported loss type\n");
            matrix_free(loss);
            return NULL;
    }
    
    matrix_set(loss, 0, 0, loss_value);
    fprintf(stderr, "Loss forward pass complete with value: %f\n", loss_value);
    return loss;
}

static matrix* loss_backward(layer* l) {
    fprintf(stderr, "Entering loss_backward...\n");
    loss_parameters* params = (loss_parameters*)l->parameters;
    
    if (!params->predicted || !params->target) {
        fprintf(stderr, "Backward pass called before forward pass\n");
        return NULL;
    }
    
    matrix* gradient = NULL;
    fprintf(stderr, "Calculating gradients for type %d\n", params->type);
    switch (params->type) {
        case LOSS_MSE:
            gradient = mse_gradient_calc(params->predicted, params->target);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            gradient = binary_cross_entropy_gradient_calc(params->predicted, params->target);
            break;
        case LOSS_CROSS_ENTROPY:
            gradient = cross_entropy_gradient_calc(params->predicted, params->target);
            break;
        default:
            fprintf(stderr, "Unsupported loss type\n");
            return NULL;
    }
    
    if (!gradient) {
        fprintf(stderr, "Failed to calculate gradients\n");
        return NULL;
    }
    
    fprintf(stderr, "Loss backward pass complete\n");
    return gradient;
}

void loss_free(layer* l) {
    fprintf(stderr, "Entering loss_free...\n");
    if (!l) {
        fprintf(stderr, "Null layer passed to loss_free\n");
        return;
    }
    
    if (l->parameters) {
        fprintf(stderr, "Freeing loss parameters...\n");
        loss_parameters* params = (loss_parameters*)l->parameters;
        if (params->predicted) {
            fprintf(stderr, "Freeing predicted matrix\n");
            matrix_free(params->predicted);
        }
        if (params->target) {
            fprintf(stderr, "Freeing target matrix\n");
            matrix_free(params->target);
        }
        free(params);
        l->parameters = NULL;
    }
    
    // Don't call layer_free here as it will cause double-free
    // Instead, free the layer structure directly
    fprintf(stderr, "Freeing layer structure\n");
    free(l);
    fprintf(stderr, "Loss free complete\n");
}
