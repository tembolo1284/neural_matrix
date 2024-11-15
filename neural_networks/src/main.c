#include "layer.h"
#include "dense_layer.h"
#include "activation.h"
#include "loss.h"
#include "optimization.h"
#include "matrix.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper function to create training data
static void create_xor_data(matrix** x_train, matrix** y_train) {
    fprintf(stderr, "Creating XOR training data...\n");
    *x_train = matrix_new(2, 4, sizeof(double));  // 2 features, 4 samples
    if (!*x_train) {
        log_error("Failed to create x_train matrix");
        exit(EXIT_FAILURE);
    }
    
    *y_train = matrix_new(1, 4, sizeof(double));  // 1 output, 4 samples
    if (!*y_train) {
        log_error("Failed to create y_train matrix");
        matrix_free(*x_train);
        exit(EXIT_FAILURE);
    }
    
    // Input data: (0,0), (0,1), (1,0), (1,1)
    double x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    memcpy((*x_train)->data, x_data, 8 * sizeof(double));
    
    // Output data: 0, 1, 1, 0
    double y_data[] = {0, 1, 1, 0};
    memcpy((*y_train)->data, y_data, 4 * sizeof(double));
    
    fprintf(stderr, "Training data created successfully\n");
}

// Train XOR network
static void train_xor_network(void) {
    log_info("Creating XOR network");
    
    // Create network layers
    fprintf(stderr, "Creating network layers...\n");
    layer* dense1 = dense_layer_new(2, 4, -1.0, 1.0);  // Input -> Hidden
    if (!dense1) {
        log_error("Failed to create dense1 layer");
        exit(EXIT_FAILURE);
    }
    
    layer* relu1 = activation_layer_new(4, ACTIVATION_RELU);
    if (!relu1) {
        log_error("Failed to create relu1 layer");
        layer_free(dense1);
        exit(EXIT_FAILURE);
    }
    
    layer* dense2 = dense_layer_new(4, 1, -1.0, 1.0);  // Hidden -> Output
    if (!dense2) {
        log_error("Failed to create dense2 layer");
        layer_free(dense1);
        layer_free(relu1);
        exit(EXIT_FAILURE);
    }
    
    layer* sigmoid = activation_layer_new(1, ACTIVATION_SIGMOID);
    if (!sigmoid) {
        log_error("Failed to create sigmoid layer");
        layer_free(dense1);
        layer_free(relu1);
        layer_free(dense2);
        exit(EXIT_FAILURE);
    }
    
    layer* loss_layer = loss_layer_new(1, LOSS_BINARY_CROSS_ENTROPY);
    if (!loss_layer) {
        log_error("Failed to create loss layer");
        layer_free(dense1);
        layer_free(relu1);
        layer_free(dense2);
        layer_free(sigmoid);
        exit(EXIT_FAILURE);
    }

    // Create optimizer with proper layer array management
    fprintf(stderr, "Creating optimizer...\n");
    layer** opt_layers = malloc(2 * sizeof(layer*));
    if (!opt_layers) {
        log_error("Failed to allocate optimizer layers array");
        layer_free(dense1);
        layer_free(relu1);
        layer_free(dense2);
        layer_free(sigmoid);
        layer_free(loss_layer);
        exit(EXIT_FAILURE);
    }
    
    opt_layers[0] = dense1;
    opt_layers[1] = dense2;
    optimizer* opt = optimizer_new(OPTIMIZER_ADAM, 0.01, 2, opt_layers);
    if (!opt) {
        log_error("Failed to create optimizer");
        free(opt_layers);
        layer_free(dense1);
        layer_free(relu1);
        layer_free(dense2);
        layer_free(sigmoid);
        layer_free(loss_layer);
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Optimizer created successfully\n");

    // Create training data
    matrix *x_train, *y_train;
    create_xor_data(&x_train, &y_train);

    // Training loop
    log_info("Training XOR network");
    for (int epoch = 0; epoch < 1000; epoch++) {
        double total_loss = 0.0;
        
        // Process each sample in the batch
        for (unsigned int i = 0; i < x_train->num_cols; i++) {
            // Extract single sample
            matrix* x_sample = matrix_new(x_train->num_rows, 1, sizeof(double));
            matrix* y_sample = matrix_new(y_train->num_rows, 1, sizeof(double));
            
            if (!x_sample || !y_sample) {
                log_error("Failed to create sample matrices");
                if (x_sample) matrix_free(x_sample);
                if (y_sample) matrix_free(y_sample);
                continue;
            }
            
            for (unsigned int r = 0; r < x_train->num_rows; r++) {
                matrix_set(x_sample, r, 0, matrix_at(x_train, r, i));
            }
            for (unsigned int r = 0; r < y_train->num_rows; r++) {
                matrix_set(y_sample, r, 0, matrix_at(y_train, r, i));
            }
            
            // Forward pass
            matrix* h1 = dense1->forward(dense1, x_sample);
            if (!h1) {
                log_error("Forward pass failed for dense1 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                continue;
            }
            
            matrix* a1 = relu1->forward(relu1, h1);
            if (!a1) {
                log_error("Forward pass failed for relu1 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                continue;
            }
            
            matrix* h2 = dense2->forward(dense2, a1);
            if (!h2) {
                log_error("Forward pass failed for dense2 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                continue;
            }
            
            matrix* output = sigmoid->forward(sigmoid, h2);
            if (!output) {
                log_error("Forward pass failed for sigmoid layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                continue;
            }
            
            matrix* loss_val = loss_layer->forward(loss_layer, output, y_sample);
            if (!loss_val) {
                log_error("Forward pass failed for loss layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                continue;
            }

            total_loss += matrix_at(loss_val, 0, 0);

            // Backward pass
            matrix* d_loss = loss_layer->backward(loss_layer, output);
            if (!d_loss) {
                log_error("Backward pass failed for loss layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                matrix_free(loss_val);
                continue;
            }
            
            matrix* d_sigmoid = sigmoid->backward(sigmoid, d_loss);
            if (!d_sigmoid) {
                log_error("Backward pass failed for sigmoid layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                matrix_free(loss_val);
                matrix_free(d_loss);
                continue;
            }
            
            matrix* d_dense2 = dense2->backward(dense2, d_sigmoid);
            if (!d_dense2) {
                log_error("Backward pass failed for dense2 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                matrix_free(loss_val);
                matrix_free(d_loss);
                matrix_free(d_sigmoid);
                continue;
            }
            
            matrix* d_relu = relu1->backward(relu1, d_dense2);
            if (!d_relu) {
                log_error("Backward pass failed for relu1 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                matrix_free(loss_val);
                matrix_free(d_loss);
                matrix_free(d_sigmoid);
                matrix_free(d_dense2);
                continue;
            }
            
            matrix* d_dense1 = dense1->backward(dense1, d_relu);
            if (!d_dense1) {
                log_error("Backward pass failed for dense1 layer");
                matrix_free(x_sample);
                matrix_free(y_sample);
                matrix_free(h1);
                matrix_free(a1);
                matrix_free(h2);
                matrix_free(output);
                matrix_free(loss_val);
                matrix_free(d_loss);
                matrix_free(d_sigmoid);
                matrix_free(d_dense2);
                matrix_free(d_relu);
                continue;
            }

            // Update parameters
            optimizer_step(opt);

            // Free intermediate matrices
            matrix_free(x_sample);
            matrix_free(y_sample);
            matrix_free(h1);
            matrix_free(a1);
            matrix_free(h2);
            matrix_free(output);
            matrix_free(loss_val);
            matrix_free(d_loss);
            matrix_free(d_sigmoid);
            matrix_free(d_dense2);
            matrix_free(d_relu);
            matrix_free(d_dense1);
        }

        if (epoch % 100 == 0) {
            log_info("Epoch %d, Average Loss: %f", epoch, total_loss / x_train->num_cols);
        }
    }

    // Test the network
    log_info("Testing XOR network");
    for (unsigned int i = 0; i < x_train->num_cols; i++) {
        matrix* x_sample = matrix_new(x_train->num_rows, 1, sizeof(double));
        if (!x_sample) {
            log_error("Failed to create test sample matrix");
            continue;
        }
        
        for (unsigned int r = 0; r < x_train->num_rows; r++) {
            matrix_set(x_sample, r, 0, matrix_at(x_train, r, i));
        }
        
        matrix* h1 = dense1->forward(dense1, x_sample);
        matrix* a1 = relu1->forward(relu1, h1);
        matrix* h2 = dense2->forward(dense2, a1);
        matrix* output = sigmoid->forward(sigmoid, h2);
        
        log_info("Input: (%f, %f) -> Output: %f", 
                matrix_at(x_sample, 0, 0),
                matrix_at(x_sample, 1, 0),
                matrix_at(output, 0, 0));

        matrix_free(x_sample);
        matrix_free(h1);
        matrix_free(a1);
        matrix_free(h2);
        matrix_free(output);
    }

    // Cleanup in correct order
    fprintf(stderr, "Starting cleanup sequence...\n");
    
    // First free the training data
    fprintf(stderr, "Freeing training data...\n");
    matrix_free(x_train);
    matrix_free(y_train);
    
    // Free the optimizer first, but prevent it from freeing the layers
    fprintf(stderr, "Freeing optimizer...\n");
    opt->num_layers = 0;  // Prevent optimizer from trying to free layers
    free(opt->layers);    // Free the layers array but not the layers themselves
    opt->layers = NULL;   // Clear the pointer
    optimizer_free(opt);  // Now free the optimizer structure
    
    // Now free the layers in reverse order of creation
    fprintf(stderr, "Freeing layers...\n");
    if (loss_layer) {
        fprintf(stderr, "Freeing loss layer...\n");
        loss_layer->free(loss_layer);
    }
    if (sigmoid) {
        fprintf(stderr, "Freeing sigmoid layer...\n");
        sigmoid->free(sigmoid);
    }
    if (dense2) {
        fprintf(stderr, "Freeing dense2 layer...\n");
        dense2->free(dense2);
    }
    if (relu1) {
        fprintf(stderr, "Freeing relu1 layer...\n");
        relu1->free(relu1);
    }
    if (dense1) {
        fprintf(stderr, "Freeing dense1 layer...\n");
        dense1->free(dense1);
    }
    
    fprintf(stderr, "Cleanup complete\n");
}

int main(void) {
    log_info("Starting neural network examples");
    
    train_xor_network();
    
    log_info("Completed neural network examples");
    return 0;
}
