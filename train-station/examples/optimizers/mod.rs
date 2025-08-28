//! Optimizer Examples for Train Station
//!
//! This module contains comprehensive examples demonstrating optimization algorithms
//! and best practices for neural network training in Train Station:
//! - Adam optimizer configurations and hyperparameters
//! - Learning rate scheduling techniques
//! - Optimizer variants comparison (Adam vs AMSGrad)
//!
//! These examples are designed to be self-contained and executable, providing
//! hands-on learning for users working with optimization in Train Station.
//!
//! # Learning Objectives
//!
//! - Understand optimizer configuration and hyperparameter tuning
//! - Learn advanced training techniques with learning rate scheduling
//! - Compare different optimizer variants for convergence analysis
//! - Implement production-ready training workflows
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see getting_started/tensor_basics.rs)
//! - Familiarity with neural network training concepts
//! - Knowledge of gradient descent optimization
//!
//! # Usage
//!
//! ```bash
//! cargo run --example adam_configurations
//! cargo run --example learning_rate_scheduling
//! ```

pub mod adam_configurations;
pub mod learning_rate_scheduling;

pub use adam_configurations::*;
pub use learning_rate_scheduling::*;
