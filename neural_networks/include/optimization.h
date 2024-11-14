#ifndef NEURAL_OPTIMIZATION_H
#define NEURAL_OPTIMIZATION_H

#include "layer.h"
#include "log.h"
#include "../matrix_lib/include/matrix.h"

typedef enum {
    OPTIMIZER_SGD,          // Stochastic Gradient Descent
    OPTIMIZER_MOMENTUM,     // SGD with Momentum
    OPTIMIZER_RMSPROP,      // RMSProp
    OPTIMIZER_ADAM          // Adam optimizer
} optimizer_type;

typedef struct {
    optimizer_type type;
    double learning_rate;
    
    // Momentum parameters
    double momentum;        // Momentum coefficient (typically 0.9)
    matrix** velocity;      // Array of velocity matrices for momentum
    
    // RMSProp parameters
    double beta2;          // Decay rate for second moment (typically 0.999)
    matrix** cache;        // Array of cache matrices for RMSProp/Adam
    
    // Adam parameters
    double beta1;          // Decay rate for first moment (typically 0.9)
    matrix** moment;       // Array of moment matrices for Adam
    unsigned long t;       // Timestep counter for Adam
    
    // Optimizer state
    unsigned int num_layers;     // Number of trainable layers
    layer** layers;             // Array of pointers to layers
} optimizer;

// Create a new optimizer
optimizer* optimizer_new(optimizer_type type, double learning_rate);

// Add a layer to be optimized
void optimizer_add_layer(optimizer* opt, layer* l);

// Initialize optimizer state (call after adding all layers)
void optimizer_init(optimizer* opt);

// Update parameters using the chosen optimization method
void optimizer_step(optimizer* opt);

// Free optimizer resources
void optimizer_free(optimizer* opt);

// Helper functions for specific optimizers
void sgd_step(optimizer* opt);
void momentum_step(optimizer* opt);
void rmsprop_step(optimizer* opt);
void adam_step(optimizer* opt);

#endif // NEURAL_OPTIMIZATION_H
