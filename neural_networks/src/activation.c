#include "activation.h"
#include "matrix.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Function prototypes for activation functions
static matrix* relu_forward(matrix* input);
static matrix* relu_backward(matrix* input, matrix* grad_output);
static matrix* sigmoid_forward(matrix* input);
static matrix* sigmoid_backward(matrix* input, matrix* grad_output);
static matrix* tanh_forward(matrix* input);
static matrix* tanh_backward(matrix* input, matrix* grad_output);
static matrix* softmax_forward(matrix* input);
static matrix* softmax_backward(matrix* input, matrix* grad_output);

// Wrapper for variadic forward propagation
static matrix* activation_forward_wrapper(layer* l, ...) {
    fprintf(stderr, "Entering activation_forward_wrapper...\n");
    va_list args;
    va_start(args, l);
    matrix* input = va_arg(args, matrix*);
    va_end(args);
    return activation_forward(l, input);
}

// Modified backward wrapper to match the layer struct's function pointer signature
static matrix* activation_backward_wrapper(layer* l, matrix* gradient) {
    fprintf(stderr, "Entering activation_backward_wrapper...\n");
    return activation_backward(l, gradient);
}

layer* activation_layer_new(unsigned int dim, activation_type type) {
    fprintf(stderr, "Creating activation layer with dim=%u and type=%d...\n", dim, type);

    layer* l = layer_new(dim, dim);
    if (!l) {
        fprintf(stderr, "Error: Failed to create layer in activation_layer_new\n");
        return NULL;
    }

    activation_parameters* params = calloc(1, sizeof(activation_parameters));
    if (!params) {
        fprintf(stderr, "Error: Failed to allocate parameters in activation_layer_new\n");
        free(l);  // Don't use layer_free here as parameters aren't set up yet
        return NULL;
    }

    params->type = type;
    params->input = NULL;
    params->output = NULL;

    l->parameters = params;
    l->forward = activation_forward_wrapper;
    l->backward = activation_backward_wrapper;
    l->free = activation_free;

    fprintf(stderr, "Activation layer created successfully\n");
    return l;
}

matrix* activation_forward(layer* l, matrix* input) {
    fprintf(stderr, "Entering activation_forward...\n");
    if (!l || !input) {
        fprintf(stderr, "Error: Null layer or input in activation_forward\n");
        return NULL;
    }

    activation_parameters* params = (activation_parameters*)l->parameters;
    if (!params) {
        fprintf(stderr, "Error: Null parameters in activation_forward\n");
        return NULL;
    }

    // Clean up previous cached input if it exists
    if (params->input) {
        fprintf(stderr, "Freeing previous cached input\n");
        matrix_free(params->input);
    }

    // Cache the input for backward pass
    params->input = matrix_copy(input);
    if (!params->input) {
        fprintf(stderr, "Error: Failed to cache input in activation_forward\n");
        return NULL;
    }

    matrix* result = NULL;
    switch (params->type) {
        case ACTIVATION_RELU:
            result = relu_forward(input);
            break;
        case ACTIVATION_SIGMOID:
            result = sigmoid_forward(input);
            break;
        case ACTIVATION_TANH:
            result = tanh_forward(input);
            break;
        case ACTIVATION_SOFTMAX:
            result = softmax_forward(input);
            break;
        default:
            fprintf(stderr, "Error: Unknown activation type in activation_forward\n");
            return NULL;
    }

    if (!result) {
        fprintf(stderr, "Error: Activation function returned NULL\n");
        return NULL;
    }

    // Cache the output
    if (params->output) {
        fprintf(stderr, "Freeing previous cached output\n");
        matrix_free(params->output);
    }
    params->output = matrix_copy(result);
    if (!params->output) {
        fprintf(stderr, "Error: Failed to cache output\n");
        matrix_free(result);
        return NULL;
    }

    fprintf(stderr, "Activation forward pass completed successfully\n");
    return result;
}

matrix* activation_backward(layer* l, matrix* gradient) {
    fprintf(stderr, "Entering activation_backward...\n");
    if (!l || !gradient) {
        fprintf(stderr, "Error: Null layer or gradient in activation_backward\n");
        return NULL;
    }

    activation_parameters* params = (activation_parameters*)l->parameters;
    if (!params || !params->input) {
        fprintf(stderr, "Error: Invalid parameters or no cached input in activation_backward\n");
        return NULL;
    }

    matrix* result = NULL;
    switch (params->type) {
        case ACTIVATION_RELU:
            result = relu_backward(params->input, gradient);
            break;
        case ACTIVATION_SIGMOID:
            result = sigmoid_backward(params->input, gradient);
            break;
        case ACTIVATION_TANH:
            result = tanh_backward(params->input, gradient);
            break;
        case ACTIVATION_SOFTMAX:
            result = softmax_backward(params->input, gradient);
            break;
        default:
            fprintf(stderr, "Error: Unknown activation type in activation_backward\n");
            return NULL;
    }

    if (!result) {
        fprintf(stderr, "Error: Activation backward pass returned NULL\n");
        return NULL;
    }

    fprintf(stderr, "Activation backward pass completed successfully\n");
    return result;
}

void activation_free(layer* l) {
    fprintf(stderr, "Entering activation_free...\n");
    if (!l) {
        fprintf(stderr, "Activation_free called with null layer\n");
        return;
    }

    if (l->parameters) {
        fprintf(stderr, "Freeing activation parameters...\n");
        activation_parameters* params = (activation_parameters*)l->parameters;
        
        if (params->input) {
            fprintf(stderr, "Freeing cached input\n");
            matrix_free(params->input);
            params->input = NULL;
        }
        
        if (params->output) {
            fprintf(stderr, "Freeing cached output\n");
            matrix_free(params->output);
            params->output = NULL;
        }
        
        fprintf(stderr, "Freeing parameters structure\n");
        free(params);
        l->parameters = NULL;
    }

    // Don't call layer_free here as it would cause recursion
    // Instead, free the layer structure directly
    fprintf(stderr, "Freeing activation layer structure\n");
    l->free = NULL;  // Prevent any potential recursive calls
    free(l);
    
    fprintf(stderr, "Activation_free completed\n");
}

static matrix* relu_forward(matrix* input) {
    fprintf(stderr, "Entering relu_forward...\n");
    if (!input) {
        fprintf(stderr, "Error: Null input in relu_forward\n");
        return NULL;
    }

    matrix* output = matrix_copy(input);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output matrix in relu_forward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, val > 0 ? val : 0);
        }
    }

    fprintf(stderr, "ReLU forward pass completed successfully\n");
    return output;
}

static matrix* relu_backward(matrix* input, matrix* grad_output) {
    fprintf(stderr, "Entering relu_backward...\n");
    if (!input || !grad_output) {
        fprintf(stderr, "Error: Null input or grad_output in relu_backward\n");
        return NULL;
    }

    matrix* grad_input = matrix_copy(grad_output);
    if (!grad_input) {
        fprintf(stderr, "Error: Failed to create gradient matrix in relu_backward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            if (matrix_at(input, i, j) <= 0) {
                matrix_set(grad_input, i, j, 0);
            }
        }
    }

    fprintf(stderr, "ReLU backward pass completed successfully\n");
    return grad_input;
}

static matrix* sigmoid_forward(matrix* input) {
    fprintf(stderr, "Entering sigmoid_forward...\n");
    if (!input) {
        fprintf(stderr, "Error: Null input in sigmoid_forward\n");
        return NULL;
    }

    matrix* output = matrix_copy(input);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output matrix in sigmoid_forward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, 1.0 / (1.0 + exp(-val)));
        }
    }

    fprintf(stderr, "Sigmoid forward pass completed successfully\n");
    return output;
}

static matrix* sigmoid_backward(matrix* input, matrix* grad_output) {
    fprintf(stderr, "Entering sigmoid_backward...\n");
    if (!input || !grad_output) {
        fprintf(stderr, "Error: Null input or grad_output in sigmoid_backward\n");
        return NULL;
    }

    matrix* grad_input = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!grad_input) {
        fprintf(stderr, "Error: Failed to create gradient matrix in sigmoid_backward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double sigmoid = 1.0 / (1.0 + exp(-matrix_at(input, i, j)));
            double grad = matrix_at(grad_output, i, j) * sigmoid * (1.0 - sigmoid);
            matrix_set(grad_input, i, j, grad);
        }
    }

    fprintf(stderr, "Sigmoid backward pass completed successfully\n");
    return grad_input;
}

static matrix* tanh_forward(matrix* input) {
    fprintf(stderr, "Entering tanh_forward...\n");
    if (!input) {
        fprintf(stderr, "Error: Null input in tanh_forward\n");
        return NULL;
    }

    matrix* output = matrix_copy(input);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output matrix in tanh_forward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double val = matrix_at(input, i, j);
            matrix_set(output, i, j, tanh(val));
        }
    }

    fprintf(stderr, "Tanh forward pass completed successfully\n");
    return output;
}

static matrix* tanh_backward(matrix* input, matrix* grad_output) {
    fprintf(stderr, "Entering tanh_backward...\n");
    if (!input || !grad_output) {
        fprintf(stderr, "Error: Null input or grad_output in tanh_backward\n");
        return NULL;
    }

    matrix* grad_input = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!grad_input) {
        fprintf(stderr, "Error: Failed to create gradient matrix in tanh_backward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double t = tanh(matrix_at(input, i, j));
            double grad = matrix_at(grad_output, i, j) * (1.0 - t * t);
            matrix_set(grad_input, i, j, grad);
        }
    }

    fprintf(stderr, "Tanh backward pass completed successfully\n");
    return grad_input;
}

static matrix* softmax_forward(matrix* input) {
    fprintf(stderr, "Entering softmax_forward...\n");
    if (!input) {
        fprintf(stderr, "Error: Null input in softmax_forward\n");
        return NULL;
    }

    matrix* output = matrix_copy(input);
    if (!output) {
        fprintf(stderr, "Error: Failed to create output matrix in softmax_forward\n");
        return NULL;
    }

    for (unsigned int i = 0; i < input->num_rows; i++) {
        // Find max in row for numerical stability
        double max_val = matrix_at(input, i, 0);
        for (unsigned int j = 1; j < input->num_cols; j++) {
            max_val = fmax(max_val, matrix_at(input, i, j));
        }

        // Compute exp and sum
        double sum = 0.0;
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double exp_val = exp(matrix_at(input, i, j) - max_val);
            matrix_set(output, i, j, exp_val);
            sum += exp_val;
        }

        // Normalize
        for (unsigned int j = 0; j < input->num_cols; j++) {
            matrix_set(output, i, j, matrix_at(output, i, j) / sum);
        }
    }

    fprintf(stderr, "Softmax forward pass completed successfully\n");
    return output;
}

static matrix* softmax_backward(matrix* input, matrix* grad_output) {
    fprintf(stderr, "Entering softmax_backward...\n");
    if (!input || !grad_output) {
        fprintf(stderr, "Error: Null input or grad_output in softmax_backward\n");
        return NULL;
    }

    matrix* grad_input = matrix_new(input->num_rows, input->num_cols, sizeof(double));
    if (!grad_input) {
        fprintf(stderr, "Error: Failed to create gradient matrix in softmax_backward\n");
        return NULL;
    }

    // Compute softmax forward pass result
    matrix* softmax_output = softmax_forward(input);
    if (!softmax_output) {
        fprintf(stderr, "Error: Failed to compute softmax in backward pass\n");
        matrix_free(grad_input);
        return NULL;
    }

    // Compute Jacobian-vector product
    for (unsigned int i = 0; i < input->num_rows; i++) {
        for (unsigned int j = 0; j < input->num_cols; j++) {
            double sum = 0.0;
            double si = matrix_at(softmax_output, i, j);
            
            for (unsigned int k = 0; k < input->num_cols; k++) {
                double sk = matrix_at(softmax_output, i, k);
                double gk = matrix_at(grad_output, i, k);
                if (j == k) {
                    sum += gk * si * (1.0 - si);
                } else {
                    sum += gk * (-si * sk);
                }
            }
            matrix_set(grad_input, i, j, sum);
        }
    }

    matrix_free(softmax_output);
    fprintf(stderr, "Softmax backward pass completed successfully\n");
    return grad_input;
}
