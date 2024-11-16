# Neural Matrix Project

A C implementation of neural networks with matrix operations, focusing on clean code and thorough testing. The project consists of two main components:

1. `matrix_lib`: A fundamental matrix operations library
2. `neural_networks`: Neural network implementation using the matrix library

## Project Structure

```
neural_matrix/

├── matrix_lib/           # Matrix operations library

│   ├── include/         # Header files

│   ├── src/            # Source files

│   └── tests/          # Matrix library tests

├── neural_networks/      # Neural network implementation

│   ├── include/         # Header files

│   ├── src/            # Source files

│   └── tests/          # Neural network tests

└── CMakeLists.txt       # Root CMake configuration
```

## Prerequisites

- C Compiler (GCC recommended)
- CMake (version 3.10 or higher)
- Ninja build system (optional but recommended)
- Criterion testing framework

### Installing Prerequisites

For Ubuntu/Pop!_OS:

```bash
# Install build essentials and CMake
sudo apt-get update
sudo apt-get install build-essential cmake ninja-build

# Install Criterion testing framework
sudo add-apt-repository ppa:snaipewastaken/ppa
sudo apt-get update
sudo apt-get install criterion-dev
```

## Building the Project

You can build the project in several ways:

### 1. Building the Entire Project

From the root directory:

```bash
# Create and enter build directory
cmake -B build -G Ninja
cmake --build build

# Run all tests
cd build && ctest
```

### 2. Building Individual Components

#### Matrix Library:

```bash
cd matrix_lib
cmake -B build -G Ninja
cmake --build build

# Run matrix library tests
cd build && ctest
```

#### Neural Networks:

```bash
cd neural_networks
cmake -B build -G Ninja
cmake --build build

# Run neural network tests
cd build && ctest
```

## Running Tests

Tests are implemented using the Criterion framework. You can run tests in several ways:

### Running All Tests

From the project root:
```bash
cd build && ctest
```

### Running Specific Test Suites

```bash
# Run matrix library tests
cd build/matrix_lib && ctest

# Run neural network tests
cd build/neural_networks && ctest
```

### Detailed Test Output

For verbose test output:
```bash
ctest --verbose
```

Or for a specific test:
```bash
ctest -R test_name --verbose
```

### Test Categories

1. Matrix Library Tests:
   - `test_matrix_init`: Matrix initialization and basic operations
   - `test_matrix_math`: Mathematical operations
   - `test_matrix_operations`: Advanced matrix operations

2. Neural Network Tests:
   - `layer_test`: Neural network layer tests
   - `dense_layer_test`: Dense layer implementation tests
   - `activation_test`: Activation function tests
   - `optimization_test`: Optimization algorithm tests
   - `loss_test`: Loss function tests

## Project Components

### Matrix Library
- Basic matrix operations (addition, subtraction, multiplication)
- Matrix initialization and memory management
- Advanced operations (transpose, element-wise operations)
- Mathematical operations (exp, log, trigonometric functions)

### Neural Networks
- Layer abstractions
- Dense (fully connected) layers
- Activation functions (ReLU, Sigmoid, Tanh, Softmax)
- Loss functions (MSE, Cross-Entropy, Binary Cross-Entropy)
- Optimization algorithms (SGD, Momentum, RMSprop, Adam)

## Using the Libraries

### Matrix Library Example
```c
#include "matrix.h"

// Create a matrix
matrix* m = matrix_new(3, 3, sizeof(double));

// Perform operations
matrix_set(m, 0, 0, 1.0);
double val = matrix_at(m, 0, 0);

// Clean up
matrix_free(m);
```

### Neural Network Example
```c
#include "layer.h"
#include "dense_layer.h"
#include "activation.h"

// Create layers
layer* dense = dense_layer_new(input_dim, output_dim, -1.0, 1.0);
layer* activation = activation_layer_new(output_dim, ACTIVATION_RELU);

// Forward pass
matrix* output = dense->forward(dense, input);
matrix* activated = activation->forward(activation, output);

// Clean up
matrix_free(output);
matrix_free(activated);
layer_free(dense);
layer_free(activation);
```

## Troubleshooting

### Common Issues

1. CMake configuration fails:
   - Ensure Criterion is properly installed
   - Check CMake version is 3.10 or higher

2. Build fails:
   - Check compiler installation
   - Ensure all dependencies are installed

3. Tests fail:
   - Run with verbose output for detailed information
   - Check Criterion installation

### Cleaning the Build

To clean and rebuild:

```bash
rm -rf build
cmake -B build -G Ninja
cmake --build build
```

