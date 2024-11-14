#ifndef NEURAL_DENSE_LAYER_H
#define NEURAL_DENSE_LAYER_H

#include "layer.h"
#include "log.h"
#include "../matrix_lib/include/matrix.h"

typedef struct {
    matrix* weights;     // Weight matrix (output_dim x input_dim)
    matrix* bias;        // Bias vector (output_dim x 1)
    matrix* input;       // Cache of input for backprop
    matrix* output;      // Cache of output for backprop
    matrix* d_weights;   // Weight gradients
    matrix* d_bias;      // Bias gradients
} dense_parameters;

// Create a new dense layer
layer* dense_layer_new(unsigned int input_dim, 
                      unsigned int output_dim, 
                      double weight_init_min, 
                      double weight_init_max);

// Forward pass (implemented internally)
matrix* dense_forward(layer* l, matrix* input);

// Backward pass (implemented internally)
matrix* dense_backward(layer* l, matrix* gradient);

// Update parameters (implemented internally)
void dense_update(layer* l, double learning_rate);

// Free dense layer resources (implemented internally)
void dense_free(layer* l);

#endif // NEURAL_DENSE_LAYER_H
