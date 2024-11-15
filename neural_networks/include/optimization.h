#ifndef NEURAL_OPTIMIZATION_H
#define NEURAL_OPTIMIZATION_H

#include "layer.h"
#include "dense_layer.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_RMSPROP,
    OPTIMIZER_ADAM
} optimizer_type;

typedef struct {
    optimizer_type type;
    double learning_rate;
    double momentum;    // For momentum and Adam
    double beta1;       // For Adam
    double beta2;       // For Adam
    double epsilon;     // For RMSprop and Adam
    unsigned int num_layers;
    layer** layers;
    matrix** velocity;  // For momentum
    matrix** cache;     // For RMSprop and Adam
    matrix** moment;    // For Adam (first moment)
    unsigned int t;     // Time step for Adam
} optimizer;

// Create a new optimizer
optimizer* optimizer_new(optimizer_type type, double learning_rate, unsigned int num_layers, layer** layers);

// Optimizer steps
void optimizer_step(optimizer* opt);
void momentum_step(optimizer* opt);
void rmsprop_step(optimizer* opt);
void adam_step(optimizer* opt);

// Clean up
void optimizer_free(optimizer* opt);

#endif // NEURAL_OPTIMIZATION_H
