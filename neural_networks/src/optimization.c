#include "optimization.h"
#include "dense_layer.h"
#include <stdlib.h>
#include <math.h>

#define EPSILON 1e-8

optimizer* optimizer_new(optimizer_type type, double learning_rate) {
    log_debug("Creating optimizer type=%d, learning_rate=%f", type, learning_rate);

    optimizer* opt = calloc(1, sizeof(optimizer));
    if (!opt) {
        log_error("Failed to allocate optimizer");
        return NULL;
    }

    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->num_layers = 0;
    opt->layers = NULL;

    // Initialize optimizer-specific parameters
    switch (type) {
        case OPTIMIZER_SGD:
            // No additional parameters needed
            break;
            
        case OPTIMIZER_MOMENTUM:
            opt->momentum = 0.9;
            opt->velocity = NULL;
            break;
            
        case OPTIMIZER_RMSPROP:
            opt->beta2 = 0.999;
            opt->cache = NULL;
            break;
            
        case OPTIMIZER_ADAM:
            opt->beta1 = 0.9;
            opt->beta2 = 0.999;
            opt->moment = NULL;
            opt->cache = NULL;
            opt->t = 0;
            break;
            
        default:
            log_error("Unknown optimizer type: %d", type);
            free(opt);
            return NULL;
    }

    log_info("Successfully created optimizer");
    return opt;
}

void optimizer_add_layer(optimizer* opt, layer* l) {
    if (!opt || !l) {
        log_error("NULL optimizer or layer in optimizer_add_layer");
        return;
    }

    // Reallocate layer array
    layer** new_layers = realloc(opt->layers, (opt->num_layers + 1) * sizeof(layer*));
    if (!new_layers) {
        log_error("Failed to reallocate layer array in optimizer");
        return;
    }

    opt->layers = new_layers;
    opt->layers[opt->num_layers] = l;
    opt->num_layers++;

    log_debug("Added layer to optimizer (total layers: %u)", opt->num_layers);
}

void optimizer_init(optimizer* opt) {
    if (!opt) {
        log_error("NULL optimizer in optimizer_init");
        return;
    }

    // Initialize optimizer-specific storage
    switch (opt->type) {
        case OPTIMIZER_MOMENTUM:
            opt->velocity = calloc(opt->num_layers, sizeof(matrix*));
            if (!opt->velocity) {
                log_error("Failed to allocate velocity matrices");
                return;
            }
            break;
            
        case OPTIMIZER_RMSPROP:
            opt->cache = calloc(opt->num_layers, sizeof(matrix*));
            if (!opt->cache) {
                log_error("Failed to allocate cache matrices");
                return;
            }
            break;
            
        case OPTIMIZER_ADAM:
            opt->moment = calloc(opt->num_layers, sizeof(matrix*));
            opt->cache = calloc(opt->num_layers, sizeof(matrix*));
            if (!opt->moment || !opt->cache) {
                log_error("Failed to allocate Adam matrices");
                free(opt->moment);
                free(opt->cache);
                return;
            }
            break;
            
        default:
            break;
    }

    log_debug("Initialized optimizer state");
}

void sgd_step(optimizer* opt) {
    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        if (l->update) {
            l->update(l, opt->learning_rate);
        }
    }
}

void momentum_step(optimizer* opt) {
    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;
        
        if (!params || !params->d_weights) continue;

        // Initialize velocity if needed
        if (!opt->velocity[i]) {
            opt->velocity[i] = matrix_new(params->d_weights->num_rows, 
                                        params->d_weights->num_cols, 
                                        sizeof(double));
        }

        // Update velocity and weights
        for (unsigned int r = 0; r < params->d_weights->num_rows; r++) {
            for (unsigned int c = 0; c < params->d_weights->num_cols; c++) {
                double v = opt->momentum * matrix_at(opt->velocity[i], r, c) -
                          opt->learning_rate * matrix_at(params->d_weights, r, c);
                matrix_set(opt->velocity[i], r, c, v);
                
                double w = matrix_at(params->weights, r, c) + v;
                matrix_set(params->weights, r, c, w);
            }
        }
    }
}

void rmsprop_step(optimizer* opt) {
    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;
        
        if (!params || !params->d_weights) continue;

        // Initialize cache if needed
        if (!opt->cache[i]) {
            opt->cache[i] = matrix_new(params->d_weights->num_rows, 
                                     params->d_weights->num_cols, 
                                     sizeof(double));
        }

        // Update cache and weights
        for (unsigned int r = 0; r < params->d_weights->num_rows; r++) {
            for (unsigned int c = 0; c < params->d_weights->num_cols; c++) {
                double g = matrix_at(params->d_weights, r, c);
                double cache = opt->beta2 * matrix_at(opt->cache[i], r, c) +
                             (1 - opt->beta2) * g * g;
                matrix_set(opt->cache[i], r, c, cache);
                
                double w = matrix_at(params->weights, r, c) -
                          opt->learning_rate * g / (sqrt(cache) + EPSILON);
                matrix_set(params->weights, r, c, w);
            }
        }
    }
}

void adam_step(optimizer* opt) {
    opt->t++;  // Increment timestep
    
    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;
        
        if (!params || !params->d_weights) continue;

        // Initialize moment and cache if needed
        if (!opt->moment[i]) {
            opt->moment[i] = matrix_new(params->d_weights->num_rows, 
                                      params->d_weights->num_cols, 
                                      sizeof(double));
        }
        if (!opt->cache[i]) {
            opt->cache[i] = matrix_new(params->d_weights->num_rows, 
                                     params->d_weights->num_cols, 
                                     sizeof(double));
        }

        // Compute bias corrections
        double bc1 = 1.0 / (1.0 - pow(opt->beta1, opt->t));
        double bc2 = 1.0 / (1.0 - pow(opt->beta2, opt->t));

        // Update moment, cache, and weights
        for (unsigned int r = 0; r < params->d_weights->num_rows; r++) {
            for (unsigned int c = 0; c < params->d_weights->num_cols; c++) {
                double g = matrix_at(params->d_weights, r, c);
                
                // Update moment (first moment)
                double m = opt->beta1 * matrix_at(opt->moment[i], r, c) +
                          (1 - opt->beta1) * g;
                matrix_set(opt->moment[i], r, c, m);
                
                // Update cache (second moment)
                double v = opt->beta2 * matrix_at(opt->cache[i], r, c) +
                          (1 - opt->beta2) * g * g;
                matrix_set(opt->cache[i], r, c, v);
                
                // Compute bias-corrected moments
                double m_hat = m * bc1;
                double v_hat = v * bc2;
                
                // Update weights
                double w = matrix_at(params->weights, r, c) -
                          opt->learning_rate * m_hat / (sqrt(v_hat) + EPSILON);
                matrix_set(params->weights, r, c, w);
            }
        }
    }
}

void optimizer_step(optimizer* opt) {
    if (!opt) {
        log_error("NULL optimizer in optimizer_step");
        return;
    }

    switch (opt->type) {
        case OPTIMIZER_SGD:
            sgd_step(opt);
            break;
        case OPTIMIZER_MOMENTUM:
            momentum_step(opt);
            break;
        case OPTIMIZER_RMSPROP:
            rmsprop_step(opt);
            break;
        case OPTIMIZER_ADAM:
            adam_step(opt);
            break;
        default:
            log_error("Unknown optimizer type: %d", opt->type);
            return;
    }

    log_debug("Completed optimizer step");
}

void optimizer_free(optimizer* opt) {
    if (!opt) {
        log_warn("Attempted to free NULL optimizer");
        return;
    }

    // Free optimizer-specific storage
    switch (opt->type) {
        case OPTIMIZER_MOMENTUM:
            if (opt->velocity) {
                for (unsigned int i = 0; i < opt->num_layers; i++) {
                    if (opt->velocity[i]) matrix_free(opt->velocity[i]);
                }
                free(opt->velocity);
            }
            break;
            
        case OPTIMIZER_RMSPROP:
            if (opt->cache) {
                for (unsigned int i = 0; i < opt->num_layers; i++) {
                    if (opt->cache[i]) matrix_free(opt->cache[i]);
                }
                free(opt->cache);
            }
            break;
            
        case OPTIMIZER_ADAM:
            if (opt->moment) {
                for (unsigned int i = 0; i < opt->num_layers; i++) {
                    if (opt->moment[i]) matrix_free(opt->moment[i]);
                }
                free(opt->moment);
            }
            if (opt->cache) {
                for (unsigned int i = 0; i < opt->num_layers; i++) {
                    if (opt->cache[i]) matrix_free(opt->cache[i]);
                }
                free(opt->cache);
            }
            break;
            
        default:
            break;
    }

    free(opt->layers);
    free(opt);

    log_debug("Freed optimizer resources");
}
