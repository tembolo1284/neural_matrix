#include "optimization.h"
#include <stdlib.h>
#include <math.h>

void optimizer_free(optimizer* opt) {
    if (!opt) return;

    if (opt->velocity) {
        for (unsigned int i = 0; i < opt->num_layers; i++) {
            if (opt->velocity[i]) matrix_free(opt->velocity[i]);
        }
        free(opt->velocity);
    }

    if (opt->cache) {
        for (unsigned int i = 0; i < opt->num_layers; i++) {
            if (opt->cache[i]) matrix_free(opt->cache[i]);
        }
        free(opt->cache);
    }

    if (opt->moment) {
        for (unsigned int i = 0; i < opt->num_layers; i++) {
            if (opt->moment[i]) matrix_free(opt->moment[i]);
        }
        free(opt->moment);
    }

    free(opt);
}

optimizer* optimizer_new(optimizer_type type, double learning_rate, unsigned int num_layers, layer** layers) {
    optimizer* opt = calloc(1, sizeof(optimizer));
    if (!opt) return NULL;

    opt->type = type;
    opt->learning_rate = learning_rate;
    opt->num_layers = num_layers;
    opt->layers = layers;

    // Set default values for hyperparameters
    opt->momentum = 0.9;      // Common default for momentum
    opt->beta1 = 0.9;         // Common default for Adam
    opt->beta2 = 0.999;       // Common default for Adam
    opt->epsilon = 1e-8;      // Small constant to prevent division by zero
    opt->t = 0;               // Initialize time step

    // Initialize arrays for optimizer states
    switch (type) {
        case OPTIMIZER_MOMENTUM:
            opt->velocity = calloc(num_layers, sizeof(matrix*));
            if (!opt->velocity) {
                optimizer_free(opt);
                return NULL;
            }
            break;

        case OPTIMIZER_RMSPROP:
            opt->cache = calloc(num_layers, sizeof(matrix*));
            if (!opt->cache) {
                optimizer_free(opt);
                return NULL;
            }
            break;

        case OPTIMIZER_ADAM:
            opt->moment = calloc(num_layers, sizeof(matrix*));
            if (!opt->moment) {
                optimizer_free(opt);
                return NULL;
            }

            opt->cache = calloc(num_layers, sizeof(matrix*));
            if (!opt->cache) {
                optimizer_free(opt);
                return NULL;
            }
            break;

        default:
            break;
    }

    return opt;
}

void optimizer_step(optimizer* opt) {
    if (!opt || !opt->layers) return;

    switch (opt->type) {
        case OPTIMIZER_SGD:
            // Basic SGD just applies gradients directly
            for (unsigned int i = 0; i < opt->num_layers; i++) {
                if (opt->layers[i] && opt->layers[i]->update) {
                    opt->layers[i]->update(opt->layers[i], opt->learning_rate);
                }
            }
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
    }
}

void momentum_step(optimizer* opt) {
    if (!opt || !opt->layers || !opt->velocity) return;

    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;

        if (!params || !params->d_weights) continue;

        // Initialize velocity if not exists
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
    if (!opt || !opt->layers || !opt->cache) return;

    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;

        if (!params || !params->d_weights) continue;

        // Initialize cache if not exists
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
                          opt->learning_rate * g / (sqrt(cache) + opt->epsilon);
                matrix_set(params->weights, r, c, w);
            }
        }
    }
}

void adam_step(optimizer* opt) {
    if (!opt || !opt->layers || !opt->moment || !opt->cache) return;

    for (unsigned int i = 0; i < opt->num_layers; i++) {
        layer* l = opt->layers[i];
        dense_parameters* params = (dense_parameters*)l->parameters;

        if (!params || !params->d_weights) continue;

        // Initialize moment and cache if not exists
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

        opt->t++; // Increment time step

        // Compute bias correction terms
        double m_correction = 1.0 / (1.0 - pow(opt->beta1, opt->t));
        double v_correction = 1.0 / (1.0 - pow(opt->beta2, opt->t));

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

                // Apply bias correction and update weights
                double m_hat = m * m_correction;
                double v_hat = v * v_correction;
                
                double w = matrix_at(params->weights, r, c) -
                          opt->learning_rate * m_hat / (sqrt(v_hat) + opt->epsilon);
                matrix_set(params->weights, r, c, w);
            }
        }
    }
}
