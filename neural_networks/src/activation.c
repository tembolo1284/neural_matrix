#include "activation.h"
#include <stdlib.h>
#include <math.h>

layer* activation_layer_new(unsigned int dim, activation_type type) {
    log_debug("Creating activation layer with dim=%u, type=%d", dim, type);

    // Create base layer (input_dim = output_dim for activation layers)
    layer* l = layer_new(dim, dim);
    if (!l) {
        log_error("Failed to create base layer for activation layer");
        return NULL;
    }

    // Allocate activation parameters
    activation_parameters* params = calloc(1, sizeof(activation_parameters));
    if (!params) {
        log_error("Failed to allocate activation parameters");
        layer_free(l);
        return NULL;
    }

    params->type = type;
    params->input = NULL;
    params->output = NULL;

    // Set layer parameters and functions
    l->parameters = params;
    l->forward = activation_forward;
    l->backward = activation_backward;
    l->update = activation_update;
    l->free = activation_free;

    log_info("Successfully created activation layer");
    return l;
}

matrix* relu_forward(matrix* input) {
    matrix* output = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate ReLU output matrix");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, val > 0 ? val : 0);
        }
    }
    return output;
}

matrix* relu_backward(matrix* input, matrix* gradient) {
    matrix* output = matrix_new(gradient->num_rows, gradient->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate ReLU gradient matrix");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, val > 0 ? matrix_at(gradient, i, j) : 0);
        }
    }
    return output;
}

matrix* sigmoid_forward(matrix* input) {
    matrix* output = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate sigmoid output matrix");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, 1.0 / (1.0 + exp(-val)));
        }
    }
    return output;
}

matrix* sigmoid_backward(matrix* input, matrix* gradient) {
    matrix* sigmoid = sigmoid_forward(input);
    if (!sigmoid) return NULL;

    matrix* output = matrix_new(gradient->num_rows, gradient->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate sigmoid gradient matrix");
        matrix_free(sigmoid);
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double s = matrix_at(sigmoid, i, j);
            matrix_set(output, i, j, matrix_at(gradient, i, j) * s * (1 - s));
        }
    }

    matrix_free(sigmoid);
    return output;
}

matrix* tanh_forward(matrix* input) {
    matrix* output = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate tanh output matrix");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, tanh(val));
        }
    }
    return output;
}

matrix* tanh_backward(matrix* input, matrix* gradient) {
    matrix* tanh_out = tanh_forward(input);
    if (!tanh_out) return NULL;

    matrix* output = matrix_new(gradient->num_rows, gradient->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate tanh gradient matrix");
        matrix_free(tanh_out);
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double t = matrix_at(tanh_out, i, j);
            matrix_set(output, i, j, matrix_at(gradient, i, j) * (1 - t * t));
        }
    }

    matrix_free(tanh_out);
    return output;
}

matrix* softmax_forward(matrix* input) {
    matrix* output = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate softmax output matrix");
        return NULL;
    }

    // Find max value in each column (for numerical stability)
    for (unsigned int j = 0; j < input->num_cols; j++) {
        double max_val = matrix_at(input, 0, j);
        for (unsigned int i = 1; i < input->num_rows; i++) {
            max_val = fmax(max_val, matrix_at(input, i, j));
        }

        // Compute exp(x - max) and sum
        double sum = 0;
        for (unsigned int i = 0; i < input->num_rows; i++) {
            double val = exp(matrix_at(input, i, j) - max_val);
            matrix_set(output, i, j, val);
            sum += val;
        }

        // Normalize
        for (unsigned int i = 0; i < input->num_rows; i++) {
            double val = matrix_at(output, i, j) / sum;
            matrix_set(output, i, j, val);
        }
    }
    return output;
}

matrix* softmax_backward(matrix* input, matrix* gradient) {
    matrix* softmax = softmax_forward(input);
    if (!softmax) return NULL;

    matrix* output = matrix_new(gradient->num_rows, gradient->num_cols, sizeof(double));
    if (!output) {
        log_error("Failed to allocate softmax gradient matrix");
        matrix_free(softmax);
        return NULL;
    }

    // For each column (each sample in the batch)
    for (unsigned int j = 0; j < input->num_cols; j++) {
        // Compute Jacobian-vector product
        for (unsigned int i = 0; i < input->num_rows; i++) {
            double sum = 0;
            for (unsigned int k = 0; k < input->num_rows; k++) {
                double s_i = matrix_at(softmax, i, j);
                double s_k = matrix_at(softmax, k, j);
                double g_k = matrix_at(gradient, k, j);
                sum += g_k * s_i * ((i == k) ? (1 - s_k) : -s_k);
            }
            matrix_set(output, i, j, sum);
        }
    }

    matrix_free(softmax);
    return output;
}

matrix* activation_forward(layer* l, matrix* input) {
    if (!l || !input) {
        log_error("NULL layer or input in activation_forward");
        return NULL;
    }

    activation_parameters* params = (activation_parameters*)l->parameters;

    // Cache input for backprop
    if (params->input) matrix_free(params->input);
    params->input = matrix_copy(input);

    // Select appropriate activation function
    matrix* output = NULL;
    switch (params->type) {
        case ACTIVATION_RELU:
            output = relu_forward(input);
            break;
        case ACTIVATION_SIGMOID:
            output = sigmoid_forward(input);
            break;
        case ACTIVATION_TANH:
            output = tanh_forward(input);
            break;
        case ACTIVATION_SOFTMAX:
            output = softmax_forward(input);
            break;
        default:
            log_error("Unknown activation type: %d", params->type);
            return NULL;
    }

    // Cache output for backprop
    if (params->output) matrix_free(params->output);
    params->output = matrix_copy(output);

    log_debug("Completed activation layer forward pass");
    return output;
}

matrix* activation_backward(layer* l, matrix* gradient) {
    if (!l || !gradient) {
        log_error("NULL layer or gradient in activation_backward");
        return NULL;
    }

    activation_parameters* params = (activation_parameters*)l->parameters;

    // Select appropriate activation gradient
    matrix* output = NULL;
    switch (params->type) {
        case ACTIVATION_RELU:
            output = relu_backward(params->input, gradient);
            break;
        case ACTIVATION_SIGMOID:
            output = sigmoid_backward(params->input, gradient);
            break;
        case ACTIVATION_TANH:
            output = tanh_backward(params->input, gradient);
            break;
        case ACTIVATION_SOFTMAX:
            output = softmax_backward(params->input, gradient);
            break;
        default:
            log_error("Unknown activation type: %d", params->type);
            return NULL;
    }

    log_debug("Completed activation layer backward pass");
    return output;
}

void activation_update(layer* l, double learning_rate) {
    (void)l;          // Unused parameter
    (void)learning_rate; // Unused parameter
    // Activation layers have no parameters to update
    log_debug("Activation layer has no parameters to update");
}

void activation_free(layer* l) {
    if (!l) {
        log_warn("Attempted to free NULL activation layer");
        return;
    }

    activation_parameters* params = (activation_parameters*)l->parameters;
    if (params) {
        if (params->input) matrix_free(params->input);
        if (params->output) matrix_free(params->output);
    }

    log_debug("Freed activation layer resources");
}
