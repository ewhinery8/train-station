//! Tensor (core): PyTorch‑inspired, zero‑dependency, maximum‑performance array with autograd
//!
//! This module contains the central `Tensor` type and its core building blocks
//! (allocation, memory, shape, operators, views). The public API is Tensor‑centric:
//! you construct and operate via `Tensor` methods; submodules exist for internal
//! organization and are referenced from the `Tensor` API.
//!
//! # Highlights
//!
//! - **Initialization**: `new`, `zeros`, `ones`, `randn`, `from_slice`, `new_on_device`,
//!   `new_uninitialized`, `new_uninitialized_aligned`
//! - **Ops**: element‑wise (+, -, *, /), scalar ops, reductions (`sum`), `matmul`
//! - **Broadcasting**: NumPy‑compatible rules for element‑wise and batched ops
//! - **Views**: zero‑copy `view` (reshape), `slice_view`, `element_view`, transpose/strides
//! - **Iterator API**: idiomatic iteration over elements/chunks/dimensions/windows that yields
//!   view tensors and preserves autograd; collect back with `collect_shape`
//! - **Autograd (GradTrack)**: thread‑safe, fast backward with `retain_grad`, `grad_owned`
//! - **Performance**: SIMD‑aligned memory, cache‑aware kernels, thread‑local memory pool
//! - **Controls**: `with_no_mem_pool` for cross‑thread ownership, `with_no_mem_padding` for exact sizes
//!
//! # Quick start
//!
//! ```
//! use train_station::Tensor;
//!
//! // Init (many options)
//! let a = Tensor::zeros(vec![2, 3]);
//! let b = Tensor::ones(vec![3]).with_requires_grad();
//! let x = Tensor::randn(vec![2, 3], None);
//!
//! // Element‑wise ops and reductions
//! let y = a.add_scalar(1.0).mul_scalar(2.0);
//! let s = y.sum();
//! assert_eq!(s.size(), 1);
//! ```
//!
//! ## Broadcasting
//!
//! ```
//! use train_station::Tensor;
//!
//! let a = Tensor::ones(vec![2, 1, 4]);
//! let b = Tensor::ones(vec![3, 1]);
//! let c = a.add_tensor(&b); // [2,1,4] + [3,1] -> [2,3,4]
//! assert_eq!(c.shape().dims(), &[2, 3, 4]);
//! ```
//!
//! ## Views (zero‑copy)
//!
//! ```
//! use train_station::Tensor;
//!
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
//! let v = x.view(vec![2, 2]);
//! assert_eq!(v.shape().dims(), &[2, 2]);
//! let e = x.element_view(2);
//! assert_eq!(e.value(), 3.0);
//! ```
//!
//! ## Iterator‑first API
//!
//! ```
//! use train_station::{Tensor, tensor::TensorCollectExt};
//!
//! let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
//! let mat = t.iter_chunks(2)
//!     .map(|chunk| chunk.mul_scalar(2.0))
//!     .collect_shape(vec![3, 2]);
//! assert_eq!(mat.shape().dims(), &[3, 2]);
//! assert_eq!(mat.data(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
//! ```
//!
//! ## Autograd (GradTrack)
//!
//! ```
//! use train_station::Tensor;
//!
//! let x = Tensor::ones(vec![2, 3]).with_requires_grad();
//! let mut loss = x.add_scalar(5.0).sum();
//! loss.backward(None);
//! let gx = x.grad_owned().unwrap();
//! assert_eq!(gx.shape().dims(), &[2, 3]);
//! ```
//!
//! ## Cross‑thread memory pool control
//!
//! ```
//! use train_station::{Tensor, tensor::with_no_mem_pool};
//! use std::thread;
//!
//! // Create in worker and return to main: prefer system allocator
//! let handle = thread::spawn(|| {
//!     with_no_mem_pool(|| Tensor::ones(vec![10]))
//! });
//! let _t = handle.join().unwrap();
//! ```
//!
//! # Memory layout & performance
//!
//! Row‑major layout with runtime‑detected SIMD alignment: typically 16/32/64‑byte alignment and
//! lane‑multiple capacity for vectorized kernels. Zero‑copy views preserve allocation ownership.

pub mod allocation;
pub mod memory;
pub mod operators;
pub mod serialization;
pub mod shape;
// Deprecated: legacy thread_pool kept only if other crates depend on it
// pub mod thread_pool;
pub mod utils;
pub mod view;

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;

use crate::device::Device;
use crate::gradtrack::engine::GraphGroupRef;
use crate::gradtrack::GradFn;

pub use allocation::Allocation;
pub use memory::{with_no_mem_pool, NoMemPoolGuard};
pub use shape::{MemoryLayout, Shape};

// Note: Prefetching functions are now in ops/add.rs where they're used

/// Global counter for unique tensor IDs
///
/// Provides thread-safe, unique identifiers for tensor gradtrack tracking.
/// Uses atomic operations to ensure uniqueness across concurrent tensor creation.
static TENSOR_ID_COUNTER: AtomicUsize = AtomicUsize::new(1);

/// High-performance multi-dimensional tensor with automatic differentiation support
///
/// The core data structure for machine learning operations, designed for maximum
/// performance with zero-cost abstractions. Supports arbitrary dimensionality,
/// SIMD optimization, gradient tracking, device placement, and natural mathematical
/// expressions through operator overloading.
///
/// # Key Features
///
/// - **Raw Pointer Storage**: Zero-overhead memory access for maximum performance
/// - **SIMD Optimization**: AVX2 alignment and vectorized operations
/// - **Memory Efficiency**: Optimized alignment strategies for different tensor sizes
/// - **gradtrack Integration**: Built-in gradient tracking and computation
/// - **Device Support**: CPU and future CUDA device placement
/// - **View Tensors**: Zero-copy tensor views with shared memory management
/// - **Thread Safety**: Send + Sync implementation for concurrent usage
/// - **Operator Overloading**: Natural mathematical expressions (+, -, *, /, +=, -=, *=, /=)
///
/// # Memory Layout
///
/// Tensors use row-major memory layout with size-dependent alignment:
/// - **Small tensors** (≤8 elements): 16-byte SSE alignment
/// - **Medium tensors** (8-1024 elements): 32-byte AVX2 alignment  
/// - **Large tensors** (>1024 elements): 64-byte cache-line alignment
///
/// # Performance Characteristics
///
/// - **Memory Overhead**: ~64 bytes per tensor (excluding data)
/// - **SIMD Ready**: Properly aligned for vectorized operations
/// - **Cache Friendly**: Optimized memory layout for CPU cache hierarchies
/// - **Zero-Cost Views**: View tensors share memory without copying
/// - **Thread Safe**: Atomic ID generation and lock-free operations
/// - **Operator Performance**: Zero-cost operator overloading for mathematical expressions
///
/// # Safety
///
/// This struct uses unsafe code for performance. The following invariants must be maintained:
/// - `data` must be valid for `shape.size` elements
/// - `data` must be properly aligned for `f32`
/// - `data` must not be aliased while the tensor exists
/// - `shape.size` must match the actual allocated memory
/// - `allocation_owner` must be valid if present
///
/// # Examples
///
/// ## Basic Tensor Operations
///
/// ```
/// use train_station::Tensor;
///
/// // Create tensors with different configurations
/// let tensor = Tensor::new(vec![2, 3]);
/// let tensor_with_grad = Tensor::ones(vec![10, 10]).with_requires_grad();
///
/// // Access tensor properties
/// assert_eq!(tensor.size(), 6);
/// assert_eq!(tensor.shape().dims(), vec![2, 3]);
/// assert!(tensor.is_contiguous());
/// ```
///
/// ## Operator Overloading
///
/// ```
/// use train_station::Tensor;
///
/// // Create tensors for operations
/// let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
/// let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
///
/// // Tensor operations with operators
/// let result = a.clone() + b.clone();                    // Tensor addition
/// let result = a.clone() * b.clone();                    // Element-wise multiplication
/// let result = a.clone() - b.clone();                    // Tensor subtraction
/// let result = a.clone() / b.clone();                    // Element-wise division
///
/// // Scalar operations
/// let result = a.clone() + 5.0;                          // Tensor + scalar
/// let result = 5.0 + a.clone();                          // Scalar + tensor
/// let result = a.clone() * 3.0;                          // Tensor * scalar
/// let result = 3.0 * a.clone();                          // Scalar * tensor
///
/// // Compound expressions
/// let result = (a.clone() + b.clone()) * 2.0 - 1.0;      // Complex mathematical expressions
///
/// // Assignment operators
/// let mut c = a.clone();
/// c += b.clone();                                        // In-place addition
/// c *= 2.0;                                              // In-place scalar multiplication
///
/// // Negation
/// let result = -a;                                       // Negate all elements
/// ```
///
/// # Thread Safety
///
/// This type is `Send + Sync` and can be safely shared between threads.
/// All operations are thread-safe through atomic ID generation and
/// thread-local gradtrack storage.
pub struct Tensor {
    /// Raw pointer to the tensor data in memory
    ///
    /// Provides zero-overhead access to tensor elements for maximum performance.
    /// The pointer is guaranteed to be valid for `shape.size` elements and properly
    /// aligned for SIMD operations. This field enables direct memory access without
    /// bounds checking overhead.
    ///
    /// # Safety
    ///
    /// - Must be valid for `shape.size` elements
    /// - Must be properly aligned for `f32` operations
    /// - Must not be aliased while tensor exists
    data: NonNull<f32>,

    /// The shape and dimensional information of the tensor
    ///
    /// Contains the dimensions, size, strides, and memory layout information.
    /// This field determines how the raw data is interpreted as a multi-dimensional
    /// tensor and enables efficient memory access patterns.
    shape: Shape,

    /// Device where this tensor is located (CPU/GPU)
    ///
    /// Determines the physical location of the tensor data and which operations
    /// can be performed on it. Currently supports CPU with future CUDA support.
    device: Device,

    /// Unique identifier for gradtrack tracking
    ///
    /// Thread-safe, globally unique ID used by the gradtrack system to track
    /// tensor operations and gradient computation. Generated atomically to
    /// ensure uniqueness across concurrent tensor creation.
    id: usize,

    /// Whether this tensor requires gradient computation
    ///
    /// Controls whether the gradtrack system tracks operations on this tensor
    /// and computes gradients during backward pass. When `true`, operations
    /// are recorded in the computation graph for gradient propagation.
    requires_grad: bool,

    /// Whether this tensor should retain its gradient after backward even if non-leaf
    ///
    /// When set via `retain_grad()`/`retain_grad_()`, users can materialize the
    /// gradient into `self.grad` after backward using `grad_or_fetch()` so that
    /// `grad()` returns `Some(&Tensor)` for non-leaf tensors too.
    retain_grad: bool,

    /// Accumulated gradients from backward pass
    ///
    /// Stores the computed gradients for this tensor after calling `backward()`.
    /// `None` if `requires_grad=false` or no gradients have been computed yet.
    /// Uses `Arc` for efficient sharing between view tensors.
    grad: Option<Arc<Tensor>>,

    /// Gradient function for gradtrack computation
    ///
    /// Records the operation that created this tensor for gradient computation
    /// during backward pass. Contains the necessary information to compute
    /// gradients with respect to input tensors.
    grad_fn: GradFn,

    /// Shared allocation owner for view tensors
    ///
    /// Enables zero-copy tensor views by sharing memory allocation between
    /// multiple tensors. `None` for tensors that own their memory directly.
    /// Uses `Arc` for thread-safe reference counting and automatic cleanup.
    allocation_owner: Option<std::sync::Arc<Allocation>>,

    /// Optional graph group reference for implicit cross-thread autograd context.
    /// None when gradients are disabled. When present, this tensor participates in
    /// the associated computation graph (local or shared).
    graph_group: Option<std::sync::Arc<GraphGroupRef>>,

    /// Phantom data to ensure proper lifetime management
    ///
    /// Ensures the tensor has the correct lifetime parameters for the `f32`
    /// data type. This prevents lifetime issues when working with raw pointers.
    _phantom: PhantomData<f32>,
}

// Make Tensor Send + Sync for thread-safe usage
//
// Safety: The raw pointer is properly managed through RAII patterns and
// the data is not shared between threads without proper synchronization.
// All tensor operations are thread-safe through atomic ID generation and
// thread-local gradtrack storage.
unsafe impl Send for Tensor {}
unsafe impl Sync for Tensor {}

// No custom Drop: memory is managed by the shared `Allocation` owner when present.

impl std::fmt::Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("size", &self.size())
            .field("id", &self.id)
            .field("requires_grad", &self.requires_grad)
            .field("has_grad", &self.grad.is_some())
            .field("has_grad_fn", &!matches!(self.grad_fn, GradFn::None))
            .finish()
    }
}

/// Clone implementation for Tensor
///
/// Creates a deep copy of the tensor data but resets gradtrack state
/// (new tensor won't track gradients unless explicitly set)
impl Clone for Tensor {
    fn clone(&self) -> Self {
        // Fast path for contiguous tensors: direct linear copy
        if self.is_contiguous() || self.size() == 0 {
            let mut cloned = Self::new(self.shape().dims().to_vec());
            unsafe {
                let src = self.as_ptr();
                let dst = cloned.as_mut_ptr();
                std::ptr::copy_nonoverlapping(src, dst, self.size());
            }
            return cloned;
        }

        // Non-contiguous view: materialize into a contiguous copy respecting strides
        let mut result = Tensor::new(self.shape().dims().to_vec());
        let rank = self.shape().rank();
        unsafe {
            let dst_ptr = result.as_mut_ptr();
            for dst_idx in 0..result.size() {
                // Compute destination coordinates under contiguous strides
                let mut coords = vec![0usize; rank];
                let mut tmp = dst_idx;
                for i in (0..rank).rev() {
                    let dim_size = self.shape().dims()[i];
                    coords[i] = tmp % dim_size;
                    tmp /= dim_size;
                }
                let src_off = self.shape().offset(&coords);
                *dst_ptr.add(dst_idx) = *self.as_ptr().add(src_off);
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    //! Core tensor functionality tests
    //!
    //! Comprehensive tests for tensor creation, memory layout, operator overloading,
    //! device management, and optimization information. Tests cover all major
    //! functionality including edge cases and performance characteristics.

    use super::*;

    /// Test basic tensor creation and properties
    ///
    /// Verifies that tensors are created with correct dimensions, size, and rank.
    /// Tests the fundamental tensor creation functionality.
    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(vec![2, 3, 4]);
        assert_eq!(tensor.size(), 24);
        assert_eq!(tensor.shape().rank(), 3);
    }

    #[test]
    fn test_tensor_1d() {
        let tensor = Tensor::new(vec![10]);
        assert_eq!(tensor.size(), 10);
        assert_eq!(tensor.shape().rank(), 1);
    }

    #[test]
    fn test_tensor_2d() {
        let tensor = Tensor::new(vec![3, 4]);
        assert_eq!(tensor.size(), 12);
        assert_eq!(tensor.shape().rank(), 2);
    }

    #[test]
    fn test_zero_sized_tensor() {
        let tensor = Tensor::new(vec![0]);
        assert_eq!(tensor.size(), 0);
    }

    #[test]
    fn test_broadcasting_compatibility() {
        let a = Tensor::new(vec![2, 3, 4]);
        let b = Tensor::new(vec![1, 3, 4]);
        let c = Tensor::new(vec![4]);
        let d = Tensor::new(vec![2, 1, 4]);
        let e = Tensor::new(vec![2, 2, 4]);

        assert!(a.is_broadcastable_with(&b));
        assert!(a.is_broadcastable_with(&c));
        assert!(a.is_broadcastable_with(&d));
        assert!(!a.is_broadcastable_with(&e)); // 3 != 2 and neither is 1
    }

    #[test]
    fn test_tensor_device_cpu() {
        use crate::device::Device;

        let tensor = Tensor::new(vec![2, 3]);
        assert_eq!(tensor.device(), Device::cpu());
        assert!(tensor.device().is_cpu());
        assert!(!tensor.device().is_cuda());
    }

    #[test]
    fn test_tensor_new_on_device_cpu() {
        use crate::device::Device;

        let tensor = Tensor::new_on_device(vec![2, 3], Device::cpu());
        assert_eq!(tensor.device(), Device::cpu());
        assert_eq!(tensor.size(), 6);
    }

    #[test]
    #[should_panic(expected = "CUDA support not enabled. Enable with --features cuda")]
    fn test_tensor_new_on_cuda_panics() {
        use crate::device::Device;

        // This should panic since CUDA feature is not enabled
        // The panic occurs when trying to create the CUDA device
        Device::cuda(0);
    }

    #[test]
    fn test_device_context_integration() {
        use crate::device::{with_device, Device};

        // Test that tensors created in different device contexts get the right device
        let tensor1 = Tensor::new(vec![2]);
        assert_eq!(tensor1.device(), Device::cpu());

        with_device(Device::cpu(), || {
            let tensor2 = Tensor::new(vec![3]);
            assert_eq!(tensor2.device(), Device::cpu());
        });
    }

    #[test]
    fn test_device_zero_sized_tensor() {
        use crate::device::Device;

        let tensor = Tensor::new_on_device(vec![0], Device::cpu());
        assert_eq!(tensor.device(), Device::cpu());
        assert_eq!(tensor.size(), 0);
    }

    /// Test data() and data_mut() methods for safe tensor data access
    #[test]
    fn test_data_access_methods() {
        // Test data() method
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let data = tensor.data();

        assert_eq!(data.len(), 4);
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);

        // Test data_mut() method
        let mut tensor = Tensor::new(vec![2, 2]);
        let data_mut = tensor.data_mut();
        data_mut[0] = 10.0;
        data_mut[1] = 20.0;
        data_mut[2] = 30.0;
        data_mut[3] = 40.0;

        // Verify changes
        assert_eq!(tensor.get(&[0, 0]), 10.0);
        assert_eq!(tensor.get(&[0, 1]), 20.0);
        assert_eq!(tensor.get(&[1, 0]), 30.0);
        assert_eq!(tensor.get(&[1, 1]), 40.0);

        // Test with zero-sized tensor
        let empty = Tensor::new(vec![0]);
        assert_eq!(empty.data().len(), 0);

        let mut empty_mut = Tensor::new(vec![0]);
        assert_eq!(empty_mut.data_mut().len(), 0);
    }

    /// Test data() method with standard library operations
    #[test]
    fn test_data_with_std_operations() {
        let tensor = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0, 5.0], vec![5]).unwrap();
        let data = tensor.data();

        // Test iterator methods
        let sum: f32 = data.iter().sum();
        assert_eq!(sum, 3.0);

        let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        assert_eq!(max, 5.0);

        let positive_count = data.iter().filter(|&&x| x > 0.0).count();
        assert_eq!(positive_count, 3);

        // Test indexing
        assert_eq!(data[0], 1.0);
        assert_eq!(data[4], 5.0);
    }

    /// Test value() method for scalar tensor access
    #[test]
    fn test_value_method() {
        // Test single-element tensor
        let scalar = Tensor::from_slice(&[42.0], vec![1]).unwrap();
        assert_eq!(scalar.value(), 42.0);

        // Test with different shapes that have size 1
        let scalar_2d = Tensor::from_slice(&[std::f32::consts::PI], vec![1, 1]).unwrap();
        assert_eq!(scalar_2d.value(), std::f32::consts::PI);

        let scalar_3d = Tensor::from_slice(&[-1.5], vec![1, 1, 1]).unwrap();
        assert_eq!(scalar_3d.value(), -1.5);

        // Test with result from iterator
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let first_elem = tensor.iter_elements().next().unwrap();
        assert_eq!(first_elem.value(), 1.0);
        assert_eq!(first_elem.shape().dims(), vec![1]);
        assert_eq!(first_elem.size(), 1);
    }

    /// Test value() method error handling
    #[test]
    #[should_panic(expected = "value() can only be called on tensors with exactly one element")]
    fn test_value_method_panics_on_multi_element() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let _ = tensor.value(); // Should panic
    }

    /// Test value() method with empty tensor
    #[test]
    #[should_panic(expected = "value() can only be called on tensors with exactly one element")]
    fn test_value_method_panics_on_empty() {
        let empty = Tensor::new(vec![0]);
        let _ = empty.value(); // Should panic
    }
}
