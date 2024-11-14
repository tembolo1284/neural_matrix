#ifndef NEURAL_ACTIVATION_H
#define NEURAL_ACTIVATION_H

#include "layer.h"
#include "log.h"
#include "../matrix_lib/include/matrix.h"

typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX
} activation_type;

typedef struct {
    activation_type type;
    matrix* input;      // Cache input for backprop
    matrix* output;     // Cache output for backprop
} activation_parameters;

// Create a new activation layer
layer* activation_layer_new(unsigned int dim, activation_type type);

// Activation functions (implemented internally)
matrix* activation_forward(layer* l, matrix* input);
matrix* activation_backward(layer* l, matrix* gradient);
void activation_update(layer* l, double learning_rate);
void activation_free(layer* l);

// Helper functions for individual activation types
matrix* relu_forward(matrix* input);
matrix* relu_backward(matrix* input, matrix* gradient);

matrix* sigmoid_forward(matrix* input);
matrix* sigmoid_backward(matrix* input, matrix* gradient);

matrix* tanh_forward(matrix* input);
matrix* tanh_backward(matrix* input, matrix* gradient);

matrix* softmax_forward(matrix* input);
matrix* softmax_backward(matrix* input, matrix* gradient);

#endif // NEURAL_ACTIVATION_H
