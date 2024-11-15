#ifndef NEURAL_DENSE_LAYER_H
#define NEURAL_DENSE_LAYER_H

#include "layer.h"
#include <stdarg.h>

typedef struct {
    matrix* weights;    // Weight matrix
    matrix* bias;       // Bias vector
    matrix* input;      // Cache input for backward pass
    matrix* output;     // Cache output for forward pass
    matrix* d_weights;  // Weight gradients
    matrix* d_bias;     // Bias gradients
} dense_parameters;

// Create a new dense layer
layer* dense_layer_new(unsigned int input_dim, unsigned int output_dim, double weight_min, double weight_max);

// Forward declarations for dense layer functions
matrix* dense_forward(layer* l, matrix* input);
matrix* dense_backward(layer* l, matrix* gradient);
void dense_update(layer* l, double learning_rate);
void dense_free(layer* l);

#endif // NEURAL_DENSE_LAYER_H
