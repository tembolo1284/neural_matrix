#ifndef NEURAL_LOSS_H
#define NEURAL_LOSS_H

#include "layer.h"
#include "log.h"
#include "matrix.h"

typedef enum {
    LOSS_MSE,           // Mean Squared Error
    LOSS_CROSS_ENTROPY, // Cross Entropy Loss (with softmax)
    LOSS_BINARY_CROSS_ENTROPY // Binary Cross Entropy
} loss_type;

typedef struct {
    loss_type type;
    matrix* predicted;  // Cache predictions for backward pass
    matrix* target;     // Cache targets for backward pass
} loss_parameters;

// Create a new loss layer
layer* loss_layer_new(unsigned int dim, loss_type type);

// Function to free loss layer
void loss_free(layer* l);

#endif // NEURAL_LOSS_H
