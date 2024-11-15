#ifndef NEURAL_ACTIVATION_H
#define NEURAL_ACTIVATION_H

#include "layer.h"
#include <stdarg.h>

typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX
} activation_type;

typedef struct {
    activation_type type;
    matrix* input;   // Cache input for backward pass
    matrix* output;  // Cache output for backward pass
} activation_parameters;

// Create a new activation layer
layer* activation_layer_new(unsigned int dim, activation_type type);

// Forward declarations for activation functions
matrix* activation_forward(layer* l, matrix* input);
matrix* activation_backward(layer* l, matrix* gradient);
void activation_free(layer* l);

#endif // NEURAL_ACTIVATION_H
