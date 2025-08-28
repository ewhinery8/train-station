//! Neural Network Examples for Train Station
//!
//! This module contains comprehensive examples demonstrating neural network building blocks
//! and complete training workflows using Train Station's tensor operations:
//! - Basic linear layer implementation with training and serialization
//! - Configurable feed-forward networks with multiple layers
//! - Linear layers and dense networks
//! - Activation functions and their properties
//! - Loss functions for different tasks
//! - Multi-layer perceptron (MLP) implementations
//! - Convolutional neural network components
//! - Transformer architecture building blocks
//!
//! These examples are designed to be self-contained and executable, providing
//! hands-on learning for users building neural networks with Train Station.
//!
//! # Learning Objectives
//!
//! - Understand neural network layer implementations using tensor operations
//! - Learn to build complete neural network architectures
//! - Explore activation functions and their mathematical properties
//! - Implement loss functions for supervised learning tasks
//! - Build and train multi-layer perceptrons
//! - Understand convolutional and transformer components
//! - Master gradient computation and backpropagation patterns
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with neural network concepts
//! - Knowledge of gradient descent and backpropagation
//! - Understanding of optimizer usage (see getting_started/optimizer_basics.rs)
//!
//! # Usage
//!
//! ```bash
//! cargo run --example basic_linear_layer
//! cargo run --example feedforward_network
//! cargo run --example linear_layers
//! cargo run --example activation_functions
//! cargo run --example loss_functions
//! cargo run --example simple_mlp
//! cargo run --example cnn_components
//! cargo run --example transformer_blocks
//! ```
//!
//! # Architecture Overview
//!
//! The examples demonstrate how to build neural networks using Train Station's
//! tensor operations, showing both the mathematical foundations and practical
//! implementation patterns:
//!
//! - **Basic Linear Layer**: Single layer implementation with training and serialization
//! - **Feed-Forward Networks**: Configurable multi-layer networks with ReLU activation
//! - **Linear Layers**: Matrix multiplication, bias addition, and parameter management
//! - **Activation Functions**: ReLU, sigmoid, tanh, and their gradient properties
//! - **Loss Functions**: MSE, cross-entropy with proper numerical stability
//! - **MLP Networks**: Complete multi-layer perceptron with training loops
//! - **CNN Components**: Convolutional layers, pooling, and feature extraction
//! - **Transformer Blocks**: Attention mechanisms and positional encoding
//!
//! # Key Concepts Demonstrated
//!
//! - **Parameter Management**: Creating and managing trainable parameters
//! - **Forward Pass**: Computing layer outputs using tensor operations
//! - **Backward Pass**: Automatic gradient computation and propagation
//! - **Loss Computation**: Implementing various loss functions
//! - **Training Loops**: Complete training workflows with optimization
//! - **Architecture Design**: Building modular and extensible networks
//! - **Numerical Stability**: Proper implementation of mathematical functions

pub mod basic_linear_layer;
pub mod feedforward_network;

pub use basic_linear_layer::*;
pub use feedforward_network::*;
