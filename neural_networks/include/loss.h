#ifndef NEURAL_LOSS_H
#define NEURAL_LOSS_H

#include "layer.h"
#include "log.h"
#include "../matrix_lib/include/matrix.h"

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

// Forward and backward functions
matrix* loss_forward(layer* l, matrix* predicted, matrix* target);
matrix* loss_backward(layer* l);
void loss_free(layer* l);

// Helper functions for individual loss types
double mse_loss(matrix* predicted, matrix* target);
matrix* mse_gradient(matrix* predicted, matrix* target);

double cross_entropy_loss(matrix* predicted, matrix* target);
matrix* cross_entropy_gradient(matrix* predicted, matrix* target);

double binary_cross_entropy_loss(matrix* predicted, matrix* target);
matrix* binary_cross_entropy_gradient(matrix* predicted, matrix* target);

#endif // NEURAL_LOSS_H
