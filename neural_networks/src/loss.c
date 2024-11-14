#include "loss.h"
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-7  // Small constant to prevent log(0)

layer* loss_layer_new(unsigned int dim, loss_type type) {
    log_debug("Creating loss layer with dim=%u, type=%d", dim, type);

    // Create base layer
    layer* l = layer_new(dim, dim);
    if (!l) {
        log_error("Failed to create base layer for loss layer");
        return NULL;
    }

    // Allocate loss parameters
    loss_parameters* params = calloc(1, sizeof(loss_parameters));
    if (!params) {
        log_error("Failed to allocate loss parameters");
        layer_free(l);
        return NULL;
    }

    params->type = type;
    params->predicted = NULL;
    params->target = NULL;

    // Set layer parameters
    l->parameters = params;

    log_info("Successfully created loss layer");
    return l;
}

double mse_loss(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in MSE loss calculation");
        return INFINITY;
    }

    if (!matrix_eqdim(predicted, target)) {
        log_error("Dimension mismatch in MSE loss calculation");
        return INFINITY;
    }

    double sum = 0.0;
    unsigned int total = predicted->num_rows * predicted->num_cols;

    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        for (unsigned int j = 0; j < predicted->num_cols; j++) {
            double diff = matrix_at(predicted, i, j) - matrix_at(target, i, j);
            sum += diff * diff;
        }
    }

    return sum / total;
}

matrix* mse_gradient(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in MSE gradient calculation");
        return NULL;
    }

    matrix* gradient = matrix_new(predicted->num_rows, predicted->num_cols, sizeof(double));
    if (!gradient) {
        log_error("Failed to allocate MSE gradient matrix");
        return NULL;
    }

    unsigned int total = predicted->num_rows * predicted->num_cols;
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        for (unsigned int j = 0; j < predicted->num_cols; j++) {
            double diff = matrix_at(predicted, i, j) - matrix_at(target, i, j);
            matrix_set(gradient, i, j, 2.0 * diff / total);
        }
    }

    return gradient;
}

double cross_entropy_loss(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in cross entropy loss calculation");
        return INFINITY;
    }

    double loss = 0.0;
    unsigned int batch_size = predicted->num_cols;

    for (unsigned int j = 0; j < predicted->num_cols; j++) {
        double sample_loss = 0.0;
        for (unsigned int i = 0; i < predicted->num_rows; i++) {
            double p = fmax(fmin(matrix_at(predicted, i, j), 1.0 - EPSILON), EPSILON);
            double t = matrix_at(target, i, j);
            sample_loss -= t * log(p);
        }
        loss += sample_loss;
    }

    return loss / batch_size;
}

matrix* cross_entropy_gradient(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in cross entropy gradient calculation");
        return NULL;
    }

    matrix* gradient = matrix_new(predicted->num_rows, predicted->num_cols, sizeof(double));
    if (!gradient) {
        log_error("Failed to allocate cross entropy gradient matrix");
        return NULL;
    }

    unsigned int batch_size = predicted->num_cols;
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        for (unsigned int j = 0; j < predicted->num_cols; j++) {
            double p = fmax(fmin(matrix_at(predicted, i, j), 1.0 - EPSILON), EPSILON);
            double t = matrix_at(target, i, j);
            matrix_set(gradient, i, j, -t / (p * batch_size));
        }
    }

    return gradient;
}

double binary_cross_entropy_loss(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in binary cross entropy loss calculation");
        return INFINITY;
    }

    double loss = 0.0;
    unsigned int total = predicted->num_rows * predicted->num_cols;

    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        for (unsigned int j = 0; j < predicted->num_cols; j++) {
            double p = fmax(fmin(matrix_at(predicted, i, j), 1.0 - EPSILON), EPSILON);
            double t = matrix_at(target, i, j);
            loss -= (t * log(p) + (1.0 - t) * log(1.0 - p));
        }
    }

    return loss / total;
}

matrix* binary_cross_entropy_gradient(matrix* predicted, matrix* target) {
    if (!predicted || !target) {
        log_error("NULL matrices in binary cross entropy gradient calculation");
        return NULL;
    }

    matrix* gradient = matrix_new(predicted->num_rows, predicted->num_cols, sizeof(double));
    if (!gradient) {
        log_error("Failed to allocate binary cross entropy gradient matrix");
        return NULL;
    }

    unsigned int total = predicted->num_rows * predicted->num_cols;
    for (unsigned int i = 0; i < predicted->num_rows; i++) {
        for (unsigned int j = 0; j < predicted->num_cols; j++) {
            double p = fmax(fmin(matrix_at(predicted, i, j), 1.0 - EPSILON), EPSILON);
            double t = matrix_at(target, i, j);
            matrix_set(gradient, i, j, (-t/p + (1-t)/(1-p)) / total);
        }
    }

    return gradient;
}

matrix* loss_forward(layer* l, matrix* predicted, matrix* target) {
    if (!l || !predicted || !target) {
        log_error("NULL arguments in loss forward pass");
        return NULL;
    }

    loss_parameters* params = (loss_parameters*)l->parameters;

    // Cache predicted and target for backward pass
    if (params->predicted) matrix_free(params->predicted);
    if (params->target) matrix_free(params->target);
    
    params->predicted = matrix_copy(predicted);
    params->target = matrix_copy(target);

    // Calculate and return the loss value as a 1x1 matrix
    matrix* loss = matrix_new(1, 1, sizeof(double));
    if (!loss) {
        log_error("Failed to allocate loss matrix");
        return NULL;
    }

    double loss_value = 0.0;
    switch (params->type) {
        case LOSS_MSE:
            loss_value = mse_loss(predicted, target);
            break;
        case LOSS_CROSS_ENTROPY:
            loss_value = cross_entropy_loss(predicted, target);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            loss_value = binary_cross_entropy_loss(predicted, target);
            break;
        default:
            log_error("Unknown loss type: %d", params->type);
            matrix_free(loss);
            return NULL;
    }

    matrix_set(loss, 0, 0, loss_value);
    log_debug("Computed loss value: %f", loss_value);
    return loss;
}

matrix* loss_backward(layer* l) {
    if (!l) {
        log_error("NULL layer in loss backward pass");
        return NULL;
    }

    loss_parameters* params = (loss_parameters*)l->parameters;
    if (!params->predicted || !params->target) {
        log_error("No cached values for loss backward pass");
        return NULL;
    }

    matrix* gradient = NULL;
    switch (params->type) {
        case LOSS_MSE:
            gradient = mse_gradient(params->predicted, params->target);
            break;
        case LOSS_CROSS_ENTROPY:
            gradient = cross_entropy_gradient(params->predicted, params->target);
            break;
        case LOSS_BINARY_CROSS_ENTROPY:
            gradient = binary_cross_entropy_gradient(params->predicted, params->target);
            break;
        default:
            log_error("Unknown loss type: %d", params->type);
            return NULL;
    }

    log_debug("Completed loss backward pass");
    return gradient;
}

void loss_free(layer* l) {
    if (!l) {
        log_warn("Attempted to free NULL loss layer");
        return;
    }

    loss_parameters* params = (loss_parameters*)l->parameters;
    if (params) {
        if (params->predicted) matrix_free(params->predicted);
        if (params->target) matrix_free(params->target);
        free(params);
    }

    log_debug("Freed loss layer resources");
}
