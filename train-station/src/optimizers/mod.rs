//! High-performance optimization algorithms for machine learning training
//!
//! This module provides a comprehensive suite of optimization algorithms designed for
//! maximum performance and compatibility with modern machine learning workflows. All
//! optimizers are implemented with zero external dependencies and feature SIMD-optimized
//! parameter updates for optimal training performance.
//!
//! # Purpose
//!
//! The optimizer module serves as the core parameter optimization layer for the Train Station
//! machine learning library, providing:
//! - **High-performance implementations**: SIMD-optimized parameter updates with AVX2 support
//! - **PyTorch compatibility**: Familiar interfaces and parameter semantics for easy migration
//! - **GradTrack integration**: Seamless integration with the automatic differentiation system
//! - **Memory efficiency**: Optimized state management with minimal memory overhead
//! - **Thread safety**: All optimizers are thread-safe and support concurrent training
//! - **Serialization support**: Complete state serialization for model checkpointing
//!
//! # Supported Optimizers
//!
//! ## Adam Optimizer
//! - **Adaptive learning rates**: Per-parameter adaptive learning rate adjustment
//! - **Momentum**: First and second moment estimation for stable convergence
//! - **Bias correction**: Proper bias correction for early training stability
//! - **AMSGrad variant**: Optional AMSGrad variant for improved convergence
//! - **Weight decay**: L2 regularization support for model regularization
//! - **SIMD optimization**: AVX2-optimized parameter updates for maximum performance
//!
//! # Design Philosophy
//!
//! ## Performance First
//! - **SIMD optimization**: All parameter updates use vectorized operations when available
//! - **Memory efficiency**: Minimal memory overhead with optimized state storage
//! - **Zero allocations**: Hot paths avoid memory allocations for maximum performance
//! - **Cache-friendly**: Memory access patterns optimized for CPU cache efficiency
//!
//! ## PyTorch Compatibility
//! - **Familiar interfaces**: Method names and semantics match PyTorch conventions
//! - **Parameter linking**: Explicit parameter registration for type safety
//! - **Learning rate scheduling**: Support for dynamic learning rate adjustment
//! - **State management**: Complete optimizer state serialization and restoration
//!
//! ## Thread Safety
//! - **Concurrent training**: All optimizers support multi-threaded parameter updates
//! - **Exclusive access**: Parameter updates require mutable references for safety
//! - **State isolation**: Each optimizer instance maintains independent state
//! - **Atomic operations**: Thread-safe operations where required
//!
//! # Usage Patterns
//!
//! ## Basic Training Loop
//! ```
//! use train_station::{Tensor, optimizers::{Adam, Optimizer}};
//!
//! // Create model parameters
//! let mut weight = Tensor::randn(vec![10, 5], None).with_requires_grad();
//! let mut bias = Tensor::zeros(vec![10]).with_requires_grad();
//!
//! // Create optimizer and link parameters
//! let mut optimizer = Adam::new();
//! optimizer.add_parameter(&weight);
//! optimizer.add_parameter(&bias);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     // Forward pass (compute loss)
//!     let input = Tensor::randn(vec![5, 3], None);
//!     let output = weight.matmul(&input);
//!     let output_with_bias = output + &bias.unsqueeze(1); // Broadcast bias to [10, 3]
//!     let target = Tensor::randn(vec![10, 3], None);
//!     let mut loss = (output_with_bias - &target).pow_scalar(2.0).sum();
//!     
//!     // Backward pass
//!     optimizer.zero_grad(&mut [&mut weight, &mut bias]);
//!     loss.backward(None);
//!     
//!     // Parameter update
//!     optimizer.step(&mut [&mut weight, &mut bias]);
//! }
//! ```
//!
//! ## Custom Configuration
//! ```
//! use train_station::optimizers::{Adam, AdamConfig, Optimizer};
//!
//! // Create custom configuration
//! let config = AdamConfig {
//!     learning_rate: 0.001,
//!     beta1: 0.9,
//!     beta2: 0.999,
//!     eps: 1e-8,
//!     weight_decay: 0.01,
//!     amsgrad: false,
//! };
//!
//! // Create optimizer with custom configuration
//! let mut optimizer = Adam::with_config(config);
//! ```
//!
//! ## State Serialization
//! ```
//! use train_station::optimizers::{Adam, Optimizer};
//! use train_station::serialization::{Serializable, Format};
//!
//! let mut optimizer = Adam::new();
//! // ... training ...
//!
//! // Save optimizer state
//! optimizer.save("optimizer.json", Format::Json).unwrap();
//!
//! // Load optimizer state
//! let mut loaded_optimizer = Adam::load("optimizer.json", Format::Json).unwrap();
//! ```
//!
//! # Performance Characteristics
//!
//! ## SIMD Optimization
//! - **AVX2 support**: Vectorized operations on x86_64 with AVX2 support
//! - **Fallback paths**: Optimized scalar implementations for non-SIMD hardware
//! - **Automatic detection**: Runtime CPU feature detection for optimal performance
//! - **Memory alignment**: Proper memory alignment for vectorized operations
//!
//! ## Memory Efficiency
//! - **Minimal overhead**: Optimized state storage with minimal memory footprint
//! - **Lazy allocation**: State allocated only when parameters are linked
//! - **Memory reuse**: Efficient memory reuse patterns to minimize allocations
//! - **Cache optimization**: Memory access patterns optimized for CPU cache
//!
//! ## Scalability
//! - **Large models**: Efficient handling of models with millions of parameters
//! - **Batch processing**: Optimized for typical machine learning batch sizes
//! - **Concurrent training**: Thread-safe operations for parallel training
//! - **Memory scaling**: Linear memory scaling with parameter count
//!
//! # Thread Safety
//!
//! All optimizers in this module are designed to be thread-safe:
//!
//! - **Exclusive access**: Parameter updates require mutable references
//! - **State isolation**: Each optimizer instance maintains independent state
//! - **Concurrent safe**: Multiple optimizers can run concurrently on different parameters
//! - **Atomic operations**: Thread-safe operations where required for correctness
//!
//! # Integration with GradTrack
//!
//! The optimizers integrate seamlessly with the GradTrack automatic differentiation system:
//!
//! - **Gradient access**: Automatic access to computed gradients from tensors
//! - **Gradient clearing**: Efficient gradient clearing before backward passes
//! - **Computation graph**: Proper integration with the computation graph system
//! - **Memory management**: Efficient gradient memory management during optimization

mod adam;

pub use adam::{Adam, AdamConfig};

/// Universal trait for parameter optimization algorithms
///
/// This trait provides a unified interface for all optimization algorithms in the Train Station
/// library, ensuring consistent behavior and API compatibility across different optimizers.
/// The trait follows PyTorch conventions for familiar usage patterns while providing
/// high-performance implementations optimized for the Train Station ecosystem.
///
/// # Design Principles
///
/// The Optimizer trait is designed around several key principles:
///
/// ## Type Safety
/// - **Parameter linking**: Explicit parameter registration prevents runtime errors
/// - **Mutable references**: Parameter updates require exclusive access for thread safety
/// - **Compile-time guarantees**: Type system ensures correct usage patterns
/// - **Memory safety**: All operations are memory-safe with proper lifetime management
///
/// ## Performance
/// - **Zero-cost abstractions**: Trait methods compile to direct function calls
/// - **SIMD optimization**: Implementations use vectorized operations when available
/// - **Memory efficiency**: Minimal overhead with optimized state management
/// - **Cache-friendly**: Memory access patterns optimized for CPU cache performance
///
/// ## PyTorch Compatibility
/// - **Familiar methods**: Method names and semantics match PyTorch conventions
/// - **Parameter management**: Similar parameter linking and state management
/// - **Learning rate control**: Dynamic learning rate adjustment support
/// - **Training workflows**: Compatible with standard training loop patterns
///
/// # Required Methods
///
/// All optimizers must implement these core methods:
///
/// * `step()` - Perform parameter updates based on current gradients
/// * `zero_grad()` - Clear accumulated gradients before backward pass
/// * `learning_rate()` - Get current learning rate for monitoring
/// * `set_learning_rate()` - Update learning rate for scheduling
///
/// # Usage Patterns
///
/// ## Basic Usage
/// ```
/// use train_station::{Tensor, optimizers::{Adam, Optimizer}};
///
/// // Create parameters and optimizer
/// let mut param = Tensor::randn(vec![10, 10], None).with_requires_grad();
/// let mut optimizer = Adam::new();
/// optimizer.add_parameter(&param);
///
/// // Training step
/// optimizer.zero_grad(&mut [&mut param]);
/// // ... forward pass and loss computation ...
/// // loss.backward(None);
/// optimizer.step(&mut [&mut param]);
/// ```
///
/// ## Learning Rate Scheduling
/// ```
/// use train_station::optimizers::{Adam, Optimizer};
///
/// let mut optimizer = Adam::new();
/// // ... parameter setup ...
///
/// for epoch in 0..100 {
///     // Decay learning rate every 10 epochs
///     if epoch % 10 == 0 {
///         let current_lr = optimizer.learning_rate();
///         optimizer.set_learning_rate(current_lr * 0.9);
///     }
///     
///     // Training step
///     // ... training logic ...
/// }
/// ```
///
/// # Thread Safety
///
/// All optimizer implementations are required to be thread-safe:
///
/// - **Send + Sync**: Optimizers can be moved between threads and shared safely
/// - **Exclusive access**: Parameter updates require mutable references
/// - **State isolation**: Each optimizer instance maintains independent state
/// - **Concurrent training**: Multiple optimizers can run concurrently
///
/// # Performance Characteristics
///
/// Optimizer implementations are expected to provide:
///
/// - **O(n) complexity**: Linear time complexity with parameter count
/// - **Minimal allocations**: Avoid memory allocations in hot paths
/// - **SIMD optimization**: Use vectorized operations when available
/// - **Cache efficiency**: Optimize memory access patterns for CPU cache
///
/// # Implementors
///
/// Current optimizer implementations:
///
/// * `Adam` - Adaptive Moment Estimation with momentum and bias correction
///
/// Future implementations may include:
/// * SGD - Stochastic Gradient Descent with momentum
/// * RMSprop - Root Mean Square Propagation
/// * AdamW - Adam with decoupled weight decay
pub trait Optimizer {
    /// Perform a single optimization step to update parameters
    ///
    /// This method performs the core optimization algorithm, updating all provided parameters
    /// based on their current gradients. The specific update rule depends on the optimizer
    /// implementation (Adam, SGD, etc.). Parameters must be linked to the optimizer before
    /// calling this method to ensure proper state management.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameter tensor references to update
    ///
    /// # Behavior
    ///
    /// The method performs these operations:
    /// 1. **Gradient validation**: Ensures all parameters have computed gradients
    /// 2. **State update**: Updates internal optimizer state (momentum, velocity, etc.)
    /// 3. **Parameter update**: Applies the optimization algorithm to update parameter values
    /// 4. **Bias correction**: Applies bias correction if required by the algorithm
    ///
    /// # Requirements
    ///
    /// - **Parameter linking**: All parameters must be linked via `add_parameter()`
    /// - **Gradient computation**: Parameters must have gradients from `backward()` call
    /// - **Exclusive access**: Requires mutable references for thread-safe updates
    /// - **Consistent state**: Optimizer state must be consistent with parameter count
    ///
    /// # Performance
    ///
    /// - **SIMD optimization**: Uses vectorized operations when available
    /// - **Memory efficiency**: Minimizes memory allocations during updates
    /// - **Cache-friendly**: Optimized memory access patterns for performance
    /// - **Linear complexity**: O(n) time complexity with parameter count
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::{Adam, Optimizer}};
    ///
    /// let mut param = Tensor::randn(vec![10, 10], None).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&param);
    ///
    /// // After forward pass and backward pass
    /// optimizer.step(&mut [&mut param]);
    /// ```
    fn step(&mut self, parameters: &mut [&mut crate::tensor::core::Tensor]);

    /// Clear accumulated gradients for all parameters
    ///
    /// This method resets all parameter gradients to zero, preparing for a new backward pass.
    /// It should be called before each backward pass to prevent gradient accumulation across
    /// multiple forward/backward cycles. This is essential for correct training behavior as
    /// gradients accumulate by default in the GradTrack system.
    ///
    /// # Arguments
    ///
    /// * `parameters` - Mutable slice of parameter tensor references to clear gradients for
    ///
    /// # Behavior
    ///
    /// The method performs these operations:
    /// 1. **Gradient clearing**: Sets all parameter gradients to zero
    /// 2. **Memory management**: Efficiently manages gradient memory allocation
    /// 3. **State consistency**: Maintains consistent gradient state across parameters
    /// 4. **GradTrack integration**: Properly integrates with the automatic differentiation system
    ///
    /// # Usage Pattern
    ///
    /// This method should be called at the beginning of each training iteration:
    /// 1. **Clear gradients**: Call `zero_grad()` to reset gradients
    /// 2. **Forward pass**: Compute model output and loss
    /// 3. **Backward pass**: Call `loss.backward()` to compute gradients
    /// 4. **Parameter update**: Call `step()` to update parameters
    ///
    /// # Performance
    ///
    /// - **Efficient clearing**: Optimized gradient clearing with minimal overhead
    /// - **Memory reuse**: Reuses existing gradient memory when possible
    /// - **SIMD optimization**: Uses vectorized operations for large parameter tensors
    /// - **Linear complexity**: O(n) time complexity with total parameter count
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, optimizers::{Adam, Optimizer}};
    ///
    /// let mut param = Tensor::randn(vec![10, 10], None).with_requires_grad();
    /// let mut optimizer = Adam::new();
    /// optimizer.add_parameter(&param);
    ///
    /// // Training iteration
    /// optimizer.zero_grad(&mut [&mut param]);  // Clear gradients
    /// // ... forward pass and loss computation ...
    /// // loss.backward(None);                   // Compute gradients
    /// optimizer.step(&mut [&mut param]);       // Update parameters
    /// ```
    ///
    /// # Integration with GradTrack
    ///
    /// The method integrates seamlessly with the GradTrack automatic differentiation system:
    /// - **Gradient storage**: Clears gradients stored in tensor gradient fields
    /// - **Computation graph**: Maintains proper computation graph state
    /// - **Memory efficiency**: Efficiently manages gradient memory allocation
    fn zero_grad(&mut self, parameters: &mut [&mut crate::tensor::core::Tensor]);

    /// Get the current learning rate for monitoring and scheduling
    ///
    /// This method returns the current learning rate used by the optimizer for parameter
    /// updates. For optimizers with adaptive learning rates, this returns the base learning
    /// rate that is modified by the adaptive algorithm. This method is essential for
    /// learning rate monitoring and implementing learning rate scheduling strategies.
    ///
    /// # Returns
    ///
    /// The current learning rate as a 32-bit floating-point value
    ///
    /// # Behavior
    ///
    /// The returned value represents:
    /// - **Base learning rate**: The configured learning rate for the optimizer
    /// - **Current rate**: The learning rate currently being used for updates
    /// - **Scheduling support**: The rate that can be modified by learning rate schedulers
    /// - **Monitoring value**: The rate that should be logged for training monitoring
    ///
    /// # Usage Patterns
    ///
    /// ## Learning Rate Monitoring
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let optimizer = Adam::new();
    /// println!("Current learning rate: {}", optimizer.learning_rate());
    /// ```
    ///
    /// ## Learning Rate Scheduling
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let mut optimizer = Adam::new();
    ///
    /// for epoch in 0..100 {
    ///     // Exponential decay every 10 epochs
    ///     if epoch % 10 == 0 && epoch > 0 {
    ///         let current_lr = optimizer.learning_rate();
    ///         optimizer.set_learning_rate(current_lr * 0.9);
    ///     }
    /// }
    /// ```
    ///
    /// ## Training Loop Integration
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let mut optimizer = Adam::new();
    ///
    /// // Training loop with learning rate logging
    /// for epoch in 0..100 {
    ///     let lr = optimizer.learning_rate();
    ///     println!("Epoch {}: Learning rate = {:.6}", epoch, lr);
    ///     
    ///     // ... training logic ...
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - **Constant time**: O(1) time complexity for learning rate retrieval
    /// - **No allocations**: No memory allocations during learning rate access
    /// - **Minimal overhead**: Negligible performance impact for monitoring
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently with other read operations.
    /// It does not modify optimizer state and can be safely used for monitoring in
    /// multi-threaded training scenarios.
    fn learning_rate(&self) -> f32;

    /// Update the learning rate for dynamic scheduling and adjustment
    ///
    /// This method updates the learning rate used by the optimizer for parameter updates.
    /// It enables dynamic learning rate adjustment during training, which is essential
    /// for implementing learning rate scheduling strategies, adaptive training, and
    /// fine-tuning workflows. The new learning rate takes effect immediately for
    /// subsequent parameter updates.
    ///
    /// # Arguments
    ///
    /// * `lr` - The new learning rate value (must be positive for meaningful optimization)
    ///
    /// # Behavior
    ///
    /// The method performs these operations:
    /// 1. **Rate validation**: Ensures the learning rate is a valid positive value
    /// 2. **State update**: Updates internal optimizer configuration with new rate
    /// 3. **Immediate effect**: New rate applies to subsequent `step()` calls
    /// 4. **Consistency**: Maintains optimizer state consistency across all parameters
    ///
    /// # Learning Rate Scheduling
    ///
    /// Common scheduling patterns supported:
    /// - **Exponential decay**: Multiply by decay factor periodically
    /// - **Step decay**: Reduce by fixed amount at specific epochs
    /// - **Cosine annealing**: Smooth cosine-based learning rate schedule
    /// - **Adaptive adjustment**: Dynamic adjustment based on training metrics
    ///
    /// # Usage Patterns
    ///
    /// ## Exponential Decay Scheduling
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let mut optimizer = Adam::new();
    ///
    /// for epoch in 0..100 {
    ///     // Decay learning rate every 10 epochs
    ///     if epoch % 10 == 0 && epoch > 0 {
    ///         let current_lr = optimizer.learning_rate();
    ///         optimizer.set_learning_rate(current_lr * 0.95);
    ///     }
    ///     
    ///     // ... training logic ...
    /// }
    /// ```
    ///
    /// ## Step-based Scheduling
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let mut optimizer = Adam::new();
    /// let initial_lr = 0.001;
    ///
    /// for epoch in 0..100 {
    ///     // Step decay at specific epochs
    ///     let lr = match epoch {
    ///         0..=29 => initial_lr,
    ///         30..=59 => initial_lr * 0.1,
    ///         60..=89 => initial_lr * 0.01,
    ///         _ => initial_lr * 0.001,
    ///     };
    ///     optimizer.set_learning_rate(lr);
    ///     
    ///     // ... training logic ...
    /// }
    /// ```
    ///
    /// ## Adaptive Adjustment
    /// ```
    /// use train_station::optimizers::{Adam, Optimizer};
    ///
    /// let mut optimizer = Adam::new();
    /// let mut best_loss = f32::INFINITY;
    /// let mut patience = 0;
    ///
    /// for epoch in 0..100 {
    ///     // ... training and validation ...
    ///     let current_loss = 0.5; // Example validation loss
    ///     
    ///     if current_loss < best_loss {
    ///         best_loss = current_loss;
    ///         patience = 0;
    ///     } else {
    ///         patience += 1;
    ///         if patience >= 5 {
    ///             // Reduce learning rate when loss plateaus
    ///             let current_lr = optimizer.learning_rate();
    ///             optimizer.set_learning_rate(current_lr * 0.5);
    ///             patience = 0;
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// # Performance
    ///
    /// - **Constant time**: O(1) time complexity for learning rate updates
    /// - **No allocations**: No memory allocations during rate updates
    /// - **Immediate effect**: Changes take effect for next parameter update
    /// - **Minimal overhead**: Negligible performance impact on training
    ///
    /// # Thread Safety
    ///
    /// This method requires exclusive access (`&mut self`) and is thread-safe when
    /// used with proper synchronization. Multiple threads should not modify the
    /// learning rate concurrently without external synchronization.
    ///
    /// # Validation
    ///
    /// While the trait does not enforce validation, implementations should:
    /// - Accept positive learning rates for normal optimization
    /// - Handle zero learning rate (effectively disables updates)
    /// - Consider very large rates that may cause numerical instability
    fn set_learning_rate(&mut self, lr: f32);
}
