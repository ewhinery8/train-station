//! High-performance automatic differentiation system for Train Station
//!
//! This module provides the core gradient tracking (GradTrack) system that enables automatic
//! differentiation for machine learning operations. The system is designed for maximum performance
//! with zero-cost abstractions and efficient memory management while maintaining mathematical
//! correctness and thread safety.
//!
//! # Purpose
//!
//! The GradTrack system serves as the foundation for automatic differentiation in Train Station,
//! providing:
//! - **Computation graph construction**: Automatic tracking of tensor operations for backpropagation
//! - **Gradient computation**: Efficient backward pass implementation with proper gradient accumulation
//! - **Memory management**: Optimized gradient storage and cleanup with minimal overhead
//! - **Thread safety**: Thread-local gradient context management for concurrent training
//! - **Performance optimization**: Zero-cost gradient function dispatch and SIMD-optimized operations
//!
//! # Core Components
//!
//! ## GradEngine
//! The central gradient computation engine that manages the computation graph and orchestrates
//! backward passes. It provides thread-local storage for gradient data and implements efficient
//! gradient accumulation algorithms.
//!
//! ## GradFn
//! Enumeration of gradient functions that represent different tensor operations. Each variant
//! contains the necessary information to compute gradients for its corresponding operation,
//! enabling zero-cost dispatch without virtual function calls.
//!
//! ## NoGradTrack
//! Context management system for disabling gradient tracking during inference or when gradients
//! are not needed. This provides significant performance improvements for forward-only operations.
//!
//! # Architecture
//!
//! The GradTrack system follows a computation graph approach similar to PyTorch's autograd:
//!
//! ## Forward Pass
//! During tensor operations, the system automatically:
//! 1. **Records operations**: Each tensor operation registers its gradient function
//! 2. **Builds computation graph**: Links between input and output tensors are established
//! 3. **Stores metadata**: Necessary information for gradient computation is preserved
//!
//! ## Backward Pass
//! When `backward()` is called:
//! 1. **Traverses graph**: Computation graph is traversed in reverse topological order
//! 2. **Computes gradients**: Each gradient function computes partial derivatives
//! 3. **Accumulates gradients**: Multiple gradients to the same tensor are properly accumulated
//! 4. **Manages memory**: Intermediate gradients are efficiently stored and cleaned up
//!
//! # Performance Characteristics
//!
//! ## Memory Efficiency
//! - **Minimal overhead**: Each tensor carries only a 1-byte GradFn enum for gradient tracking
//! - **Efficient storage**: Thread-local gradient storage minimizes allocation overhead
//! - **Smart cleanup**: Automatic gradient cleanup prevents memory leaks
//!
//! ## Computational Efficiency
//! - **Zero-cost dispatch**: Enum-based gradient functions eliminate virtual function overhead
//! - **SIMD optimization**: Gradient computations leverage vectorized operations where possible
//! - **Lazy evaluation**: Gradients are computed only when needed during backward pass
//!
//! ## Thread Safety
//! - **Thread-local storage**: Each thread maintains its own computation graph
//! - **No global state**: Eliminates synchronization overhead in multi-threaded training
//! - **Context isolation**: Gradient contexts are properly isolated between threads
//!
//! # Integration with Tensor Operations
//!
//! The GradTrack system integrates seamlessly with tensor operations:
//! - **Automatic registration**: Tensor operations automatically register gradient functions
//! - **Transparent operation**: No changes needed to existing tensor operation code
//! - **Conditional tracking**: Gradient tracking can be enabled/disabled per tensor
//! - **Efficient propagation**: Gradients flow efficiently through complex computation graphs
//!
//! # Thread Safety
//!
//! All components in this module are designed to be thread-safe:
//! - **Thread-local gradient storage**: Each thread maintains independent gradient state
//! - **Atomic gradient context**: Gradient enable/disable state is managed atomically
//! - **Safe concurrent access**: Multiple threads can perform gradient operations simultaneously
//! - **No data races**: Careful design eliminates potential data races in gradient computation

/// Gradient computation engine and computation graph management
///
/// This module contains the core gradient engine that manages computation graphs,
/// orchestrates backward passes, and handles gradient accumulation. It provides
/// thread-local storage for gradient data and implements efficient algorithms
/// for gradient computation and memory management.
pub mod engine;

/// Gradient function enumeration and dispatch system
///
/// This module defines the GradFn enum that represents different tensor operations
/// and their corresponding gradient computation functions. It enables zero-cost
/// gradient function dispatch without virtual function overhead while maintaining
/// type safety and performance.
pub mod grad_fn;

/// Gradient context management for inference optimization
///
/// This module provides context management for disabling gradient tracking when
/// gradients are not needed, such as during inference or evaluation. It offers
/// significant performance improvements by eliminating gradient computation overhead
/// for forward-only operations.
pub mod no_grad_track;

// Re-exports for convenient access to core gradient tracking functionality

/// Clear all accumulated gradients from the current thread's gradient storage
///
/// This function removes all gradient data from the thread-local gradient storage,
/// effectively resetting the gradient state. It's useful for cleaning up after
/// training iterations or when switching between different computation contexts.
pub use engine::clear_gradients;

/// Retrieve accumulated gradient for a specific tensor
///
/// This function returns the accumulated gradient for a tensor identified by its
/// unique ID. It provides access to the final gradient values after backward
/// pass completion, enabling gradient inspection and custom gradient processing.
pub use engine::get_accumulated_gradient;

/// The central gradient computation engine
///
/// GradEngine orchestrates the entire gradient computation process, managing
/// computation graphs, coordinating backward passes, and handling gradient
/// accumulation. It provides the primary interface for gradient-related operations
/// in the Train Station automatic differentiation system.
pub use engine::GradEngine;

/// Enumeration of gradient functions for different tensor operations
///
/// GradFn represents the gradient computation logic for various tensor operations,
/// enabling efficient dispatch and gradient computation. Each variant contains
/// the necessary metadata to compute gradients for its corresponding operation
/// while maintaining zero-cost abstraction principles.
pub use grad_fn::GradFn;

/// Check if gradient tracking is currently enabled
///
/// This function returns the current gradient tracking state for the calling thread.
/// It's useful for conditional logic that depends on whether gradients are being
/// computed, allowing for performance optimizations in gradient-aware code.
pub use no_grad_track::is_grad_enabled;

/// Enable or disable gradient tracking for the current thread
///
/// This function allows manual control over gradient tracking state, enabling
/// fine-grained control over when gradients are computed. It's particularly
/// useful for implementing custom training loops or inference optimizations.
pub use no_grad_track::set_grad_enabled;

/// Execute a closure with gradient tracking temporarily disabled
///
/// This function provides a convenient way to execute code without gradient
/// tracking, automatically restoring the previous gradient state when the
/// closure completes. It's ideal for inference operations or temporary
/// gradient-free computations within training code.
pub use no_grad_track::with_no_grad;

/// RAII guard for temporarily disabling gradient tracking
///
/// NoGradTrack provides a scope-based mechanism for disabling gradient tracking,
/// automatically restoring the previous state when the guard is dropped. It
/// ensures proper gradient state management even in the presence of early
/// returns or exceptions.
pub use no_grad_track::NoGradTrack;
