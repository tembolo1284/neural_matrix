#ifndef NEURAL_LAYER_H
#define NEURAL_LAYER_H

#include "log.h"
#include "../matrix_lib/include/matrix.h"

typedef struct layer_s {
    unsigned int input_dim;
    unsigned int output_dim;
    
    // Forward pass
    matrix* (*forward)(struct layer_s* layer, matrix* input);
    
    // Backward pass
    matrix* (*backward)(struct layer_s* layer, matrix* gradient);
    
    // Update parameters
    void (*update)(struct layer_s* layer, double learning_rate);
    
    // Free layer resources
    void (*free)(struct layer_s* layer);

    // Layer-specific data
    void* parameters;
} layer;

// Layer initialization
layer* layer_new(unsigned int input_dim, unsigned int output_dim);

// Layer operations
matrix* layer_forward(layer* l, matrix* input);
matrix* layer_backward(layer* l, matrix* gradient);
void layer_update(layer* l, double learning_rate);
void layer_free(layer* l);

#endif // NEURAL_LAYER_H
