#include "neural_networks/include/layer.h"
#include "neural_networks/include/dense_layer.h"
#include "neural_networks/include/activation.h"
#include "neural_networks/include/loss.h"
#include "neural_networks/include/optimization.h"
#include "matrix_lib/include/matrix.h"
#include "log.h"
#include <stdio.h>
#include <stdlib.h>

// Helper function to create training data
void create_xor_data(matrix** x_train, matrix** y_train) {
    *x_train = matrix_new(2, 4, sizeof(double));  // 2 features, 4 samples
    *y_train = matrix_new(1, 4, sizeof(double));  // 1 output, 4 samples
    
    // Input data: (0,0), (0,1), (1,0), (1,1)
    double x_data[] = {0, 0, 0, 1, 1, 0, 1, 1};
    memcpy((*x_train)->data, x_data, 8 * sizeof(double));
    
    // Output data: 0, 1, 1, 0
    double y_data[] = {0, 1, 1, 0};
    memcpy((*y_train)->data, y_data, 4 * sizeof(double));
}

// Example 1: XOR network demonstration
void train_xor_network(void) {
    log_info("Creating XOR network");

    // Create network layers
    layer* dense1 = dense_layer_new(2, 4, -1.0, 1.0);  // Input -> Hidden
    layer* relu1 = activation_layer_new(4, ACTIVATION_RELU);
    layer* dense2 = dense_layer_new(4, 1, -1.0, 1.0);  // Hidden -> Output
    layer* sigmoid = activation_layer_new(1, ACTIVATION_SIGMOID);
    layer* loss_layer = loss_layer_new(1, LOSS_BINARY_CROSS_ENTROPY);

    // Create optimizer
    optimizer* opt = optimizer_new(OPTIMIZER_ADAM, 0.01);
    optimizer_add_layer(opt, dense1);
    optimizer_add_layer(opt, dense2);
    optimizer_init(opt);

    // Create training data
    matrix *x_train, *y_train;
    create_xor_data(&x_train, &y_train);

    // Training loop
    log_info("Training XOR network");
    for (int epoch = 0; epoch < 1000; epoch++) {
        // Forward pass
        matrix* h1 = dense1->forward(dense1, x_train);
        matrix* a1 = relu1->forward(relu1, h1);
        matrix* h2 = dense2->forward(dense2, a1);
        matrix* output = sigmoid->forward(sigmoid, h2);
        matrix* loss_val = loss_layer->forward(loss_layer, output, y_train);

        if (epoch % 100 == 0) {
            log_info("Epoch %d, Loss: %f", epoch, matrix_at(loss_val, 0, 0));
        }

        // Backward pass
        matrix* d_loss = loss_layer->backward(loss_layer);
        matrix* d_sigmoid = sigmoid->backward(sigmoid, d_loss);
        matrix* d_dense2 = dense2->backward(dense2, d_sigmoid);
        matrix* d_relu = relu1->backward(relu1, d_dense2);
        matrix* d_dense1 = dense1->backward(dense1, d_relu);

        // Update parameters
        optimizer_step(opt);

        // Free intermediate matrices
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

    // Test the network
    log_info("Testing XOR network");
    matrix* test_input = x_train;  // Use training data as test
    matrix* h1 = dense1->forward(dense1, test_input);
    matrix* a1 = relu1->forward(relu1, h1);
    matrix* h2 = dense2->forward(dense2, a1);
    matrix* output = sigmoid->forward(sigmoid, h2);

    // Print results
    for (unsigned int i = 0; i < test_input->num_cols; i++) {
        double x1 = matrix_at(test_input, 0, i);
        double x2 = matrix_at(test_input, 1, i);
        double pred = matrix_at(output, 0, i);
        log_info("Input: (%f, %f) -> Output: %f", x1, x2, pred);
    }

    // Cleanup
    matrix_free(h1);
    matrix_free(a1);
    matrix_free(h2);
    matrix_free(output);
    matrix_free(x_train);
    matrix_free(y_train);
    layer_free(dense1);
    layer_free(relu1);
    layer_free(dense2);
    layer_free(sigmoid);
    layer_free(loss_layer);
    optimizer_free(opt);
}

// Example 2: Simple function approximation network
void train_regression_network(void) {
    log_info("Creating regression network");

    // Create network layers
    layer* dense1 = dense_layer_new(1, 8, -0.5, 0.5);   // Input -> Hidden
    layer* tanh1 = activation_layer_new(8, ACTIVATION_TANH);
    layer* dense2 = dense_layer_new(8, 1, -0.5, 0.5);   // Hidden -> Output
    layer* loss_layer = loss_layer_new(1, LOSS_MSE);

    // Create optimizer
    optimizer* opt = optimizer_new(OPTIMIZER_ADAM, 0.01);
    optimizer_add_layer(opt, dense1);
    optimizer_add_layer(opt, dense2);
    optimizer_init(opt);

    // Create training data (approximating y = x^2)
    matrix* x_train = matrix_new(1, 100, sizeof(double));
    matrix* y_train = matrix_new(1, 100, sizeof(double));
    
    for (int i = 0; i < 100; i++) {
        double x = (double)i / 50.0 - 1.0;  // [-1, 1]
        matrix_set(x_train, 0, i, x);
        matrix_set(y_train, 0, i, x * x);   // y = x^2
    }

    // Training loop
    log_info("Training regression network");
    for (int epoch = 0; epoch < 1000; epoch++) {
        // Forward pass
        matrix* h1 = dense1->forward(dense1, x_train);
        matrix* a1 = tanh1->forward(tanh1, h1);
        matrix* output = dense2->forward(dense2, a1);
        matrix* loss_val = loss_layer->forward(loss_layer, output, y_train);

        if (epoch % 100 == 0) {
            log_info("Epoch %d, Loss: %f", epoch, matrix_at(loss_val, 0, 0));
        }

        // Backward pass
        matrix* d_loss = loss_layer->backward(loss_layer);
        matrix* d_dense2 = dense2->backward(dense2, d_loss);
        matrix* d_tanh = tanh1->backward(tanh1, d_dense2);
        matrix* d_dense1 = dense1->backward(dense1, d_tanh);

        // Update parameters
        optimizer_step(opt);

        // Free intermediate matrices
        matrix_free(h1);
        matrix_free(a1);
        matrix_free(output);
        matrix_free(loss_val);
        matrix_free(d_loss);
        matrix_free(d_dense2);
        matrix_free(d_tanh);
        matrix_free(d_dense1);
    }

    // Test the network
    log_info("Testing regression network");
    double test_points[] = {-0.8, -0.4, 0.0, 0.4, 0.8};
    for (int i = 0; i < 5; i++) {
        matrix* x = matrix_new(1, 1, sizeof(double));
        matrix_set(x, 0, 0, test_points[i]);
        
        matrix* h1 = dense1->forward(dense1, x);
        matrix* a1 = tanh1->forward(tanh1, h1);
        matrix* output = dense2->forward(dense2, a1);
        
        log_info("x: %f, predicted y: %f, actual y: %f", 
                test_points[i], 
                matrix_at(output, 0, 0), 
                test_points[i] * test_points[i]);

        matrix_free(x);
        matrix_free(h1);
        matrix_free(a1);
        matrix_free(output);
    }

    // Cleanup
    matrix_free(x_train);
    matrix_free(y_train);
    layer_free(dense1);
    layer_free(tanh1);
    layer_free(dense2);
    layer_free(loss_layer);
    optimizer_free(opt);
}

int main(void) {
    log_info("Starting neural network examples");
    
    // Example 1: XOR Network
    train_xor_network();
    
    // Example 2: Regression Network
    train_regression_network();
    
    log_info("Completed neural network examples");
    return 0;
}
