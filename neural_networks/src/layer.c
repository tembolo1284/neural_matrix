#include "layer.h"
#include "log.h"
#include <stdlib.h>

layer* layer_new(unsigned int input_dim, unsigned int output_dim) {
    if (input_dim == 0 || output_dim == 0) {
        log_error("Invalid dimensions: input_dim=%u, output_dim=%u", input_dim, output_dim);
        return NULL;
    }

    layer* l = calloc(1, sizeof(layer));
    if (!l) {
        log_error("Failed to allocate memory for layer");
        return NULL;
    }

    l->input_dim = input_dim;
    l->output_dim = output_dim;
    
    // Initialize function pointers to NULL
    l->forward = NULL;
    l->backward = NULL;
    l->update = NULL;
    l->free = NULL;
    l->parameters = NULL;

    log_debug("Created new layer with input_dim=%u, output_dim=%u", input_dim, output_dim);
    return l;
}

matrix* layer_forward(layer* l, matrix* input) {
    if (!l || !input) {
        log_error("NULL layer or input in forward pass");
        return NULL;
    }

    if (!l->forward) {
        log_error("Forward function not implemented for layer");
        return NULL;
    }

    if (input->num_rows != l->input_dim) {
        log_error("Input dimension mismatch: expected %u, got %u", 
                 l->input_dim, input->num_rows);
        return NULL;
    }

    log_debug("Executing forward pass for layer");
    return l->forward(l, input);
}

matrix* layer_backward(layer* l, matrix* gradient) {
    if (!l || !gradient) {
        log_error("NULL layer or gradient in backward pass");
        return NULL;
    }

    if (!l->backward) {
        log_error("Backward function not implemented for layer");
        return NULL;
    }

    if (gradient->num_rows != l->output_dim) {
        log_error("Gradient dimension mismatch: expected %u, got %u", 
                 l->output_dim, gradient->num_rows);
        return NULL;
    }

    log_debug("Executing backward pass for layer");
    return l->backward(l, gradient);
}

void layer_update(layer* l, double learning_rate) {
    if (!l) {
        log_error("NULL layer in update");
        return;
    }

    if (!l->update) {
        log_error("Update function not implemented for layer");
        return;
    }

    if (learning_rate <= 0) {
        log_warn("Learning rate is <= 0: %f", learning_rate);
    }

    log_debug("Updating layer parameters with learning rate %f", learning_rate);
    l->update(l, learning_rate);
}

void layer_free(layer* l) {
    if (!l) {
        log_warn("Attempted to free NULL layer");
        return;
    }

    if (l->free) {
        log_debug("Executing layer-specific cleanup");
        l->free(l);
    }

    if (l->parameters) {
        log_debug("Freeing layer parameters");
        free(l->parameters);
    }

    log_debug("Freeing layer structure");
    free(l);
}
