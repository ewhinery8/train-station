//! High-performance automatic differentiation (GradTrack)
//!
//! Concise, PyTorch-inspired gradient tracking for all `Tensor` ops (including broadcasting,
//! views, iterators, and matmul). Designed for research and performance:
//! - **Transparent**: ops auto-register gradients; call `backward()` on a result
//! - **Thread-safe**: local graphs for single-thread speed; shared graphs unify across threads
//! - **Efficient**: zero-cost dispatch for gradient functions; vectorized math where possible
//! - **Practical hygiene**: explicit clearing helpers to avoid stale tensors lingering on graphs
//!
//! ## Quick examples
//!
//! Basic backward and gradient access (owned):
//! ```
//! use train_station::Tensor;
//!
//! let x = Tensor::ones(vec![2, 3]).with_requires_grad();
//! let y = x.mul_scalar(2.0).add_scalar(1.0);
//! let mut loss = y.sum();
//! loss.backward(None);
//! let gx = x.grad_owned().unwrap();
//! assert_eq!(gx.shape().dims(), vec![2, 3]);
//! ```
//!
//! Disable gradients for inference:
//! ```
//! use train_station::gradtrack::{with_no_grad, NoGradTrack};
//! use train_station::Tensor;
//!
//! let a = Tensor::ones(vec![4, 4]).with_requires_grad();
//! with_no_grad(|| {
//!     let _z = a.mul_scalar(3.0); // no grad tracked
//! });
//!
//! let _guard = NoGradTrack::new();
//! let _z2 = a.add_scalar(5.0); // also no grad tracked in this scope
//! ```
//!
//! Clearing to prevent stale tensors on graphs:
//! ```
//! use train_station::gradtrack::{clear_gradient_for_tensor, clear_local_graph, clear_all_shared_graphs};
//! use train_station::Tensor;
//!
//! let x = Tensor::ones(vec![2, 2]).with_requires_grad();
//! let mut s = x.sum();
//! s.backward(None);
//! clear_gradient_for_tensor(x.id());
//! clear_local_graph();
//! // If using shared graphs across threads in your app:
//! clear_all_shared_graphs();
//! ```
//!
//! Cross-thread creation and returning tensors to main:
//! ```
//! use train_station::Tensor;
//! use train_station::tensor::with_no_mem_pool; // system allocator for returned tensors
//! use std::thread;
//!
//! // Create a tensor in a worker and return it to main
//! let handle = thread::spawn(|| {
//!     with_no_mem_pool(|| Tensor::ones(vec![8]).with_requires_grad())
//! });
//! let t = handle.join().unwrap();
//! // Continue building the graph on main thread
//! let y = t.mul_scalar(2.0);
//! let mut loss = y.sum();
//! loss.backward(None);
//! assert!(t.grad_owned().is_some());
//! ```
//!
//! Tip: GradTrack cooperates with all major initialization methods (`zeros`, `ones`, `randn`,
//! `from_slice`, `new`), element-wise ops, activations, broadcasting, matmul, and view/iterator-based
//! transforms.
//!
//! ## Troubleshooting
//!
//! - Gradients are stale or unexpectedly persist across iterations:
//!   - Call `clear_gradient_for_tensor(id)` for specific tensors, or `clear_local_graph()` / `clear_all_shared_graphs()`
//!     at iteration boundaries in long-running services.
//! - No gradient returned:
//!   - Ensure inputs have `.with_requires_grad()` and `backward(None)` was called on a leaf result.
//!   - Use `grad_owned()` when you need ownership, `grad()` for a reference. For non-leaf tensors using
//!     `retain_grad()`, call `materialize_grad()` or `grad_or_fetch()` after backward.
//! - Cross-thread usage:
//!   - If creating tensors in worker threads and returning them to main, wrap creation in
//!     `train_station::tensor::with_no_mem_pool(|| ...)` so allocations use the system allocator
//!     instead of a thread-local pool.

/// Gradient computation engine and computation graph management
///
/// This module contains the core gradient engine that manages computation graphs,
/// orchestrates backward passes, and handles gradient accumulation. It provides
/// thread-local storage for gradient data and implements efficient algorithms
/// for gradient computation and memory management.
pub(crate) mod engine;

/// Gradient function enumeration and dispatch system
///
/// This module defines the GradFn enum that represents different tensor operations
/// and their corresponding gradient computation functions. It enables zero-cost
/// gradient function dispatch without virtual function overhead while maintaining
/// type safety and performance.
pub(crate) mod grad_fn;

/// Gradient context management for inference optimization
///
/// This module provides context management for disabling gradient tracking when
/// gradients are not needed, such as during inference or evaluation. It offers
/// significant performance improvements by eliminating gradient computation overhead
/// for forward-only operations.
pub(crate) mod no_grad_track;

/// Retrieve accumulated gradient for a specific tensor
///
/// This function returns the accumulated gradient for a tensor identified by its
/// unique ID. It provides access to the final gradient values after backward
/// pass completion, enabling gradient inspection and custom gradient processing.
pub(crate) use engine::get_accumulated_gradient;

/// The central gradient computation engine
///
/// GradEngine orchestrates the entire gradient computation process, managing
/// computation graphs, coordinating backward passes, and handling gradient
/// accumulation. It provides the primary interface for gradient-related operations
/// in the Train Station automatic differentiation system.
pub(crate) use engine::GradEngine;
pub use engine::{
    clear_all_graphs_known, clear_all_shared_graphs, clear_gradient_for_tensor, clear_gradients,
    clear_graph_for_tensor, clear_local_graph, clear_shared_graph_for_tensor,
};

/// Enumeration of gradient functions for different tensor operations
///
/// GradFn represents the gradient computation logic for various tensor operations,
/// enabling efficient dispatch and gradient computation. Each variant contains
/// the necessary metadata to compute gradients for its corresponding operation
/// while maintaining zero-cost abstraction principles.
pub(crate) use grad_fn::GradFn;

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
