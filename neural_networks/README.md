# Neural Networks in C

A neural network implementation in C focusing on modularity and educational purposes. Built on top of a custom matrix library for all linear algebra operations.

## Features

- Modular layer architecture
- Multiple layer types:
 - Dense (Fully Connected) layers
 - Various activation layers (ReLU, Sigmoid, Tanh, Softmax)
- Different loss functions:
 - Mean Squared Error (MSE)
 - Binary Cross Entropy
 - Categorical Cross Entropy
- Multiple optimizers:
 - Stochastic Gradient Descent (SGD)
 - Momentum
 - RMSProp
 - Adam
- Example implementations:
 - XOR problem (Binary Classification)
 - Function approximation (Regression)

## Prerequisites

- CMake (>= 3.10)
- Ninja build system
- C compiler (gcc/clang)
- Criterion (for tests)

### Installing Prerequisites on Ubuntu/Debian

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build
sudo apt install libcriterion-dev
```

# Building the Project

## Configure

```bash
cmake -B build -G Ninja
```

## Build

```bash
cmake --build build
```

## Running Tests

### Run all tests with failure output:

```bash
ctest --test-dir build --output-on-failure
```

## Run specific test categories:

# Run only layer tests
ctest --test-dir build --output-on-failure -R layer

# Run only optimizer tests
ctest --test-dir build --output-on-failure -R optimizer

# Running Examples

## After building, run the examples:

```bash
./build/examples/neural_networks_examples
```
This will run two demonstration networks:

1. XOR Problem - Shows binary classification

2. Function Approximation - Shows regression capabilities


# Project Structure

neural_networks/

├── CMakeLists.txt

├── include/

│   ├── layer.h

│   ├── dense_layer.h

│   ├── activation.h

│   ├── loss.h

│   └── optimization.h

├── src/

│   ├── layer.c

│   ├── dense_layer.c

│   ├── activation.c

│   ├── loss.c

│   └── optimization.c

└── tests/

    ├── layer_test.c

    ├── dense_layer_test.c

    ├── activation_test.c

    ├── loss_test.c

    └── optimization_test.c



