//! Tensor utility functions and core implementation methods
//!
//! This module provides essential utility functions for tensor creation, memory management,
//! gradient tracking, and optimization. It contains the core implementation methods
//! that enable efficient tensor operations and gradtrack functionality.
//!
//! # Key Features
//!
//! - **Tensor Creation**: Optimized constructors with memory alignment
//! - **Memory Management**: Safe memory access and allocation utilities
//! - **Gradient Tracking**: GradTrack system integration and gradient management
//! - **Performance Optimization**: SIMD-ready memory layout and alignment
//! - **Device Management**: CPU and future CUDA device support
//! - **Memory Layout**: Contiguous, strided, and view memory access patterns
//!
//! # Performance Characteristics
//!
//! - **Memory Alignment**: 16-byte SSE, 32-byte AVX2, 64-byte cache-line alignment
//! - **SIMD Optimization**: Properly aligned memory for vectorized operations
//! - **Zero-Cost Abstractions**: Minimal overhead for utility operations
//! - **Thread Safety**: Atomic operations for gradient tracking and ID generation
//! - **Memory Efficiency**: Optimized allocation strategies for different tensor sizes
//!
//! # Examples
//!
//! ## Basic Tensor Creation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors of different sizes
//! let small_tensor = Tensor::new(vec![2, 3]);      // 16-byte alignment
//! let medium_tensor = Tensor::new(vec![32, 32]);   // 32-byte alignment
//! let large_tensor = Tensor::new(vec![1000, 1000]); // 64-byte alignment
//!
//! // Initialize data before use
//! let mut tensor = Tensor::new(vec![2, 3]);
//! tensor.fill(0.0); // Initialize with zeros
//! ```
//!
//! ## Gradient Tracking
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
//! assert!(tensor.requires_grad());
//! ```
//!
//! ## Memory Access
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let value = tensor.get(&[0, 1]);
//! assert_eq!(value, 2.0);
//!
//! let mut tensor = Tensor::new(vec![2, 2]);
//! tensor.set(&[0, 1], 42.0);
//! assert_eq!(tensor.get(&[0, 1]), 42.0);
//! ```

use std::alloc::Layout;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use crate::device::current_device;
use crate::gradtrack::engine::ensure_local_group_for_tensor;
use crate::gradtrack::{self, GradEngine, GradFn};
#[cfg(target_arch = "x86_64")]
#[cfg(test)]
use crate::tensor::core::memory::SimdLevel;
use crate::tensor::core::memory::{compute_allocation_params, use_pool_alloc_enabled};
use crate::tensor::core::{Allocation, Device, TENSOR_ID_COUNTER};
use crate::tensor::Shape;

use super::Tensor;

impl Tensor {
    /// Ensures that this tensor has unique ownership of its allocation before mutation.
    ///
    /// If the underlying allocation is shared (i.e., multiple views), this performs
    /// a copy-on-write: allocates a new buffer and copies data so that mutating this tensor
    /// does not affect existing views.
    fn ensure_unique_allocation(&mut self) {
        if self.size() == 0 {
            return;
        }
        if let Some(owner) = self.allocation_owner.as_ref() {
            if std::sync::Arc::strong_count(owner) > 1 {
                // Allocate a new buffer with the same size/alignment policy as constructors
                let (alignment, padded_elems) = compute_allocation_params(self.shape.size());
                let total_size = padded_elems * std::mem::size_of::<f32>();
                let layout = Layout::from_size_align(total_size, alignment)
                    .expect("Failed to create layout for tensor data");

                let alloc_obj = if use_pool_alloc_enabled() {
                    Allocation::new_pooled(padded_elems, alignment, layout)
                } else {
                    Allocation::new_uninitialized(padded_elems, alignment, layout)
                };

                let new_ptr = alloc_obj.ptr;
                unsafe {
                    std::ptr::copy_nonoverlapping(self.as_ptr(), new_ptr.as_ptr(), self.size());
                }

                // Replace pointer and owner with the new unique allocation
                self.data = new_ptr;
                self.allocation_owner = Some(std::sync::Arc::new(alloc_obj));
            }
        }
    }
    // ===== Runtime SIMD capability helpers for ops selection =====

    /// Returns the highest runtime-detected SIMD level on this CPU
    #[inline]
    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn simd_runtime_level() -> SimdLevel {
        use crate::tensor::core::memory::detect_runtime_simd;

        detect_runtime_simd()
    }

    /// Returns the SIMD lane width (elements per vector) for f32 at runtime
    #[inline]
    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn simd_lane_width_elems_runtime() -> usize {
        use crate::tensor::core::memory::simd_lane_width_elems;

        simd_lane_width_elems(Self::simd_runtime_level())
    }

    /// Checks whether this tensor's data pointer is aligned for the specified SIMD level
    #[inline]
    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn is_aligned_for_level(&self, level: SimdLevel) -> bool {
        use crate::tensor::core::memory::simd_alignment_bytes;

        let required = simd_alignment_bytes(level);
        (self.data.as_ptr() as usize).is_multiple_of(required)
    }

    /// Returns true if this tensor is aligned for the current runtime SIMD level
    #[inline]
    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn is_aligned_for_runtime_level(&self) -> bool {
        self.is_aligned_for_level(Self::simd_runtime_level())
    }

    /// Returns true if ops should use aligned SIMD loads/stores without a tail branch
    /// (i.e., pointer is aligned for the runtime SIMD level and length is a multiple of lane)
    #[inline]
    #[cfg(test)]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn prefer_aligned_simd_ops(&self) -> bool {
        let lane = Self::simd_lane_width_elems_runtime();
        self.is_aligned_for_runtime_level() && self.size().is_multiple_of(lane)
    }

    /// Returns true if ops should use unaligned SIMD loads/stores (mm_loadu)
    /// because pointer alignment or length does not meet aligned requirements.
    /// Returns false when no SIMD is available.
    #[inline]
    #[cfg(target_arch = "x86_64")]
    #[cfg(test)]
    pub(crate) fn should_use_unaligned_simd_ops(&self) -> bool {
        match Self::simd_runtime_level() {
            SimdLevel::Scalar => false,
            _ => !self.prefer_aligned_simd_ops(),
        }
    }

    /// Returns the allocated capacity in elements, which may be padded beyond logical size
    #[inline]
    pub fn capacity_elems(&self) -> usize {
        if let Some(owner) = self.allocation_owner() {
            owner.capacity_elems()
        } else {
            self.size()
        }
    }
    /// Creates a new tensor with the specified shape and optimized memory layout
    ///
    /// Allocates memory with size-dependent alignment for optimal performance:
    /// - Small tensors (≤8 elements): 16-byte SSE alignment
    /// - Medium tensors (8-1024 elements): 32-byte AVX2 alignment
    /// - Large tensors (>1024 elements): 64-byte cache-line alignment
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    ///
    /// # Returns
    ///
    /// A new tensor with uninitialized data. The data must be initialized
    /// before use to avoid undefined behavior.
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Single allocation with optimized alignment
    /// - **SIMD Ready**: Properly aligned for vectorized operations
    /// - **Cache Friendly**: Optimized for CPU cache hierarchies
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Safety
    ///
    /// The returned tensor contains uninitialized memory. You must initialize
    /// the data before performing any operations that read from it.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create tensors of different sizes
    /// let small_tensor = Tensor::new(vec![2, 3]);      // 16-byte alignment
    /// let medium_tensor = Tensor::new(vec![32, 32]);   // 32-byte alignment
    /// let large_tensor = Tensor::new(vec![1000, 1000]); // 64-byte alignment
    ///
    /// // Initialize data before use
    /// let mut tensor = Tensor::new(vec![2, 3]);
    /// tensor.fill(0.0); // Initialize with zeros
    /// ```
    #[inline]
    #[track_caller]
    pub fn new(shape_dims: Vec<usize>) -> Self {
        let shape = Shape::new(shape_dims);
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if shape.size() == 0 {
            // Handle zero-sized tensors
            return Self {
                data: NonNull::dangling(),
                shape,
                device: current_device(),
                id,
                requires_grad: false,
                retain_grad: false,
                grad: None,
                grad_fn: GradFn::None,
                allocation_owner: None,
                graph_group: None,
                _phantom: PhantomData,
            };
        }

        // Compute alignment and padded element count based on runtime SIMD
        let (alignment, padded_elems) = compute_allocation_params(shape.size());
        let total_size = padded_elems * std::mem::size_of::<f32>();

        let layout = Layout::from_size_align(total_size, alignment)
            .expect("Failed to create layout for tensor data");

        // Allocation policy: prefer pool for tiny tensors and medium/large classes.
        // Route certain small sizes to system allocator to keep pool stats semantics used by tests.
        let alloc_obj = if use_pool_alloc_enabled() {
            Allocation::new_pooled(padded_elems, alignment, layout)
        } else {
            Allocation::new(padded_elems, alignment, layout)
        };
        let ptr = alloc_obj.ptr;

        debug_assert!(
            alloc_obj.capacity_elems() >= padded_elems,
            "Allocation capacity ({}) smaller than padded elements ({})",
            alloc_obj.capacity_elems(),
            padded_elems
        );

        Self {
            data: ptr,
            shape,
            device: current_device(),
            id,
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner: Some(std::sync::Arc::new(alloc_obj)),
            graph_group: None,
            _phantom: PhantomData,
        }
    }

    /// Returns the shape and dimensional information of the tensor
    ///
    /// Provides access to the tensor's dimensions, size, strides, and memory
    /// layout information. This is used for shape validation, memory access
    /// calculations, and optimization decisions.
    ///
    /// # Returns
    ///
    /// Reference to the tensor's shape information containing dimensions,
    /// size, strides, and memory layout type.
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - direct field access
    /// - **Memory**: No allocation - returns reference to existing data
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// let shape = tensor.shape();
    /// assert_eq!(shape.dims(), vec![2, 3, 4]);
    /// assert_eq!(shape.size(), 24);
    /// assert_eq!(shape.rank(), 3);
    /// ```
    #[inline]
    #[track_caller]
    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the total number of elements in the tensor
    ///
    /// Provides the total count of elements across all dimensions. This is
    /// used for memory allocation, iteration bounds, and performance optimization.
    ///
    /// # Returns
    ///
    /// Total number of elements as `usize`
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - direct field access
    /// - **Memory**: No allocation - returns stored value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// assert_eq!(tensor.size(), 24); // 2 * 3 * 4
    ///
    /// let scalar = Tensor::new(vec![1]);
    /// assert_eq!(scalar.size(), 1);
    ///
    /// let empty = Tensor::new(vec![0]);
    /// assert_eq!(empty.size(), 0);
    /// ```
    #[inline]
    #[track_caller]
    pub fn size(&self) -> usize {
        self.shape().size()
    }

    /// Returns the device where this tensor is located
    ///
    /// Provides the physical location of the tensor data (CPU/GPU). This
    /// determines which operations can be performed on the tensor and where
    /// computations will be executed.
    ///
    /// # Returns
    ///
    /// Device enum indicating the tensor's physical location
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - direct field access
    /// - **Memory**: No allocation - returns stored value
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3]);
    /// assert!(tensor.device().is_cpu());
    /// assert!(!tensor.device().is_cuda());
    /// ```
    #[inline]
    #[track_caller]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Creates a new tensor with the specified shape on a specific device
    ///
    /// Allocates memory on the specified device with the same optimized alignment
    /// strategy as `new()`. Currently supports CPU device with future CUDA support.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    /// * `device` - The device where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A new tensor with uninitialized data on the specified device
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Device-specific allocation with optimized alignment
    /// - **SIMD Ready**: Properly aligned for vectorized operations on target device
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Panics
    ///
    /// Panics if the specified device is not supported (e.g., CUDA without feature flag)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new_on_device(vec![2, 3], train_station::Device::cpu());
    /// assert!(tensor.device().is_cpu());
    /// assert_eq!(tensor.size(), 6);
    /// ```
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    /// * `device` - Device where the tensor should be allocated
    ///
    /// # Returns
    ///
    /// A new tensor with uninitialized data on the specified device
    ///
    /// # Panics
    ///
    /// Panics if the device is not supported (currently only CPU is supported)
    ///
    /// # Performance
    ///
    /// - **Memory Allocation**: Single allocation with optimized alignment
    /// - **SIMD Ready**: Properly aligned for vectorized operations
    /// - **Cache Friendly**: Optimized for CPU cache hierarchies
    /// - **Thread Safe**: Atomic ID generation for gradtrack tracking
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::{Tensor, Device};
    ///
    /// // Create tensor on CPU device
    /// let tensor = Tensor::new_on_device(vec![2, 3], Device::cpu());
    /// assert_eq!(tensor.device(), Device::cpu());
    /// assert_eq!(tensor.size(), 6);
    /// ```
    #[track_caller]
    pub fn new_on_device(shape_dims: Vec<usize>, device: Device) -> Self {
        // For now, only CPU is supported
        if !device.is_cpu() {
            panic!("Only CPU device is currently supported. CUDA support is planned for future releases.");
        }

        let shape = Shape::new(shape_dims);
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if shape.size() == 0 {
            // Handle zero-sized tensors
            return Self {
                data: NonNull::dangling(),
                shape,
                device,
                id,
                requires_grad: false,
                retain_grad: false,
                grad: None,
                grad_fn: GradFn::None,
                allocation_owner: None,
                graph_group: None,
                _phantom: PhantomData,
            };
        }

        let (alignment, padded_elems) = compute_allocation_params(shape.size());
        let total_size = padded_elems * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(total_size, alignment)
            .expect("Failed to create layout for tensor data");

        // Same allocation policy as new(): prefer pool for tiny and medium/large classes
        let alloc_obj = if use_pool_alloc_enabled() {
            Allocation::new_pooled(padded_elems, alignment, layout)
        } else {
            Allocation::new(padded_elems, alignment, layout)
        };
        let ptr = alloc_obj.ptr;

        debug_assert!(
            alloc_obj.capacity_elems() >= padded_elems,
            "Allocation capacity ({}) smaller than padded elements ({})",
            alloc_obj.capacity_elems(),
            padded_elems
        );

        Self {
            data: ptr,
            shape,
            device,
            id,
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner: Some(std::sync::Arc::new(alloc_obj)),
            graph_group: None,
            _phantom: PhantomData,
        }
    }

    /// Enable gradient computation for this tensor
    ///
    /// Builder method that enables automatic gradient tracking for this tensor.
    /// When enabled, all operations involving this tensor will be recorded in
    /// the computation graph for gradient computation during backward pass.
    ///
    /// # Returns
    ///
    /// `self` with gradient tracking enabled
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - simple field assignment
    /// - **Memory**: No additional allocation
    /// - **Overhead**: Minimal gradtrack tracking overhead when gradients computed
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// assert!(tensor.requires_grad());
    /// ```
    #[track_caller]
    pub fn with_requires_grad(mut self) -> Self {
        self.requires_grad = true;
        if self.graph_group.is_none() {
            self.graph_group = Some(ensure_local_group_for_tensor(self.id()));
        }
        self
    }

    /// Set gradient tracking for this tensor
    ///
    /// Controls whether the gradtrack system tracks operations on this tensor
    /// and computes gradients during backward pass. When disabled, clears
    /// any existing gradients and gradient functions.
    ///
    /// # Arguments
    ///
    /// * `requires_grad` - Whether to track gradients for this tensor
    ///
    /// # Performance
    ///
    /// - **Time Complexity**: O(1) - simple field assignment
    /// - **Memory**: May free gradient storage when disabled
    /// - **Overhead**: Zero overhead when gradients disabled
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::ones(vec![2, 3]);
    /// tensor.set_requires_grad(true);
    /// assert!(tensor.requires_grad());
    ///
    /// // Disable gradient tracking
    /// tensor.set_requires_grad(false);
    /// assert!(!tensor.requires_grad());
    /// ```
    #[track_caller]
    pub fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self.grad = None;
            self.grad_fn = GradFn::None;
            self.graph_group = None;
        } else {
            // Lazily bind to a local graph group for this tensor id
            if self.graph_group.is_none() {
                self.graph_group = Some(ensure_local_group_for_tensor(self.id()));
            }
        }
    }

    /// Mark this tensor to retain gradients after backward, even if it is non-leaf.
    ///
    /// Builder-style API: returns self with `retain_grad=true`.
    /// Call `materialize_grad()` or `grad_or_fetch()` after backward to copy the
    /// accumulated gradient from the GradGraph into `self.grad` so `grad()` works.
    #[track_caller]
    pub fn retain_grad(mut self) -> Self {
        self.retain_grad = true;
        // Register intent with grad engine to keep gradient available after backward
        crate::gradtrack::engine::mark_retain_grad(self.id);
        self
    }

    /// In-place variant to enable or disable gradient retention for non-leaf tensors
    #[track_caller]
    pub fn retain_grad_(&mut self, enable: bool) {
        self.retain_grad = enable;
        if enable {
            crate::gradtrack::engine::mark_retain_grad(self.id);
        }
    }

    /// Check if this tensor requires gradients
    ///
    /// # Returns
    ///
    /// `true` if gradient tracking is enabled for this tensor
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3]);
    /// assert!(!tensor.requires_grad());
    ///
    /// let grad_tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// assert!(grad_tensor.requires_grad());
    /// ```
    #[track_caller]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    /// Get a reference to this tensor's locally cached gradient (if any)
    ///
    /// This accessor returns only the gradient cached on this tensor's `grad` field.
    /// It does NOT query the global/autograd gradient storage. For leaf tensors,
    /// gradients are accumulated and tracked by the grad engine and are not
    /// automatically written back to the local `grad` field.
    ///
    /// To make `grad()` return `Some(&Tensor)`:
    /// - Enable retention on this tensor (typically for non-leaf tensors) with
    ///   `retain_grad_(&mut tensor, true)` or `tensor.retain_grad()`
    /// - After `backward()`, call `tensor.materialize_grad()` or
    ///   `tensor.grad_or_fetch()` to copy the accumulated gradient from the
    ///   autograd engine into `self.grad`
    ///
    /// If you want to read gradients without caching them locally, prefer
    /// [`grad_owned`](#method.grad_owned), which consults the global gradient
    /// storage.
    ///
    /// # Returns
    ///
    /// Optional reference to the locally cached gradient tensor, or `None` if
    /// not materialized on this tensor.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut x = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut loss = x.sum();
    /// loss.backward(None);
    /// // Without materialization, grad() typically returns None for leaves
    /// assert!(x.grad().is_none());
    /// // Materialize to cache locally so grad() works
    /// x.retain_grad_ (true);
    /// x.materialize_grad();
    /// assert!(x.grad().is_some());
    /// ```
    #[track_caller]
    pub fn grad(&self) -> Option<&Tensor> {
        // First check if we have a gradient stored directly
        if let Some(grad) = self.grad.as_ref() {
            return Some(grad.as_ref());
        }

        None
    }

    /// Fetch the accumulated gradient after backward and cache it on this tensor if `retain_grad` is enabled.
    ///
    /// Returns true if a gradient was found and cached. After a successful call,
    /// `grad()` will return `Some(&Tensor)` even for non-leaf tensors.
    #[track_caller]
    pub fn materialize_grad(&mut self) -> bool {
        if self.grad.is_some() {
            return true;
        }
        if !self.retain_grad {
            return false;
        }
        if let Some(g) = self.grad_owned() {
            self.set_grad(g);
            return true;
        }
        false
    }

    /// Convenience accessor: if `retain_grad` is enabled, fetch and cache the gradient
    /// on first access so callers can immediately get a reference.
    #[track_caller]
    pub fn grad_or_fetch(&mut self) -> Option<&Tensor> {
        if self.grad.is_none() && self.retain_grad {
            if let Some(g) = self.grad_owned() {
                self.set_grad(g);
            }
        }
        self.grad.as_deref()
    }

    /// Get the accumulated gradient from the autograd engine as an owned tensor
    ///
    /// This accessor queries the global/shared gradient storage for this tensor's
    /// ID and returns the current accumulated gradient by value. It complements
    /// [`grad`](#method.grad), which only returns a locally cached reference.
    ///
    /// - Works for leaf tensors without any prior materialization step
    /// - Does not modify or clear internal gradient state
    /// - Suitable for optimizers and logging that need the latest gradients
    ///
    /// # Returns
    ///
    /// `Some(Tensor)` containing the current accumulated gradient when available,
    /// otherwise `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut x = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut loss = x.sum();
    /// loss.backward(None);
    ///
    /// // Fetch directly from autograd storage (no materialization required)
    /// let g = x.grad_owned().unwrap();
    /// assert_eq!(g.shape().dims(), vec![2, 3]);
    /// ```
    #[track_caller]
    pub fn grad_owned(&self) -> Option<Tensor> {
        // First check if we have a gradient stored directly
        if let Some(grad) = self.grad.as_ref() {
            return Some((**grad).clone());
        }

        // Always consult the global/shared gradient storage so gradients accumulated
        // in a promoted/merged shared group are visible regardless of the calling thread.
        // This is critical for cross-thread forward/backward validation and parity with LibTorch.
        use crate::gradtrack;
        gradtrack::get_accumulated_gradient(self.id)
    }

    /// Get the unique ID of this tensor
    ///
    /// Returns the unique identifier assigned to this tensor during creation.
    /// This ID is used for gradtrack tracking and tensor identification.
    ///
    /// # Returns
    ///
    /// Unique tensor ID as `usize`
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor1 = Tensor::new(vec![2, 3]);
    /// let tensor2 = Tensor::new(vec![2, 3]);
    /// assert_ne!(tensor1.id(), tensor2.id()); // Each tensor has unique ID
    /// ```
    #[track_caller]
    pub fn id(&self) -> usize {
        self.id
    }

    /// Detach this tensor from the computation graph
    ///
    /// Returns a new tensor with the same data but no gradient tracking.
    /// This is useful when you want to use a tensor in inference without
    /// affecting the computation graph.
    ///
    /// # Returns
    ///
    /// A new tensor with the same data but gradient tracking disabled
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let detached = tensor.detach();
    /// assert!(!detached.requires_grad());
    /// assert_eq!(tensor.size(), detached.size());
    /// ```
    #[track_caller]
    pub fn detach(&self) -> Self {
        let mut detached = Self::new(self.shape().dims().to_vec());

        // Copy data
        unsafe {
            let src = self.as_ptr();
            let dst = detached.as_mut_ptr();
            std::ptr::copy_nonoverlapping(src, dst, self.size());
        }

        detached
    }

    /// Create a new tensor that doesn't track gradients from this one
    ///
    /// Similar to detach() but modifies this tensor in place. This is useful
    /// when you want to disable gradient tracking for the current tensor
    /// without creating a copy.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// assert!(tensor.requires_grad());
    /// tensor.detach_();
    /// assert!(!tensor.requires_grad());
    /// ```
    #[track_caller]
    pub fn detach_(&mut self) {
        self.requires_grad = false;
        self.grad = None;
        self.grad_fn = GradFn::None;
    }

    /// Entry point for backward pass on this tensor
    ///
    /// Computes gradients for all tensors in the computation graph that have
    /// `requires_grad` set to true. This is the main entry point for automatic
    /// differentiation.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Optional gradient tensor for the output. If None, assumes
    ///   the tensor is a scalar (e.g., loss value) and uses a tensor of ones.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// let mut result = tensor.add_scalar(5.0);
    /// result.backward(None);
    /// // Note: Gradient computation depends on the gradtrack system implementation
    /// ```
    #[track_caller]
    pub fn backward(&mut self, grad_output: Option<Tensor>) {
        GradEngine::backward(self, grad_output);
    }

    /// Returns a raw pointer to the tensor data for unsafe operations
    ///
    /// # Safety
    ///
    /// This is unsafe because it provides direct access to the underlying memory.
    /// The caller must ensure:
    /// - The tensor is not dropped while the pointer is used
    /// - No concurrent mutable access occurs
    /// - Bounds are respected
    #[inline]
    pub unsafe fn as_ptr(&self) -> *const f32 {
        self.data.as_ptr()
    }

    /// Returns a mutable raw pointer to the tensor data for unsafe operations
    ///
    /// # Safety
    ///
    /// This is unsafe because it provides direct mutable access to the underlying memory.
    /// The caller must ensure:
    /// - The tensor is not dropped while the pointer is used
    /// - No concurrent access occurs
    /// - Bounds are respected
    #[inline]
    pub unsafe fn as_mut_ptr(&mut self) -> *mut f32 {
        self.data.as_ptr()
    }

    /// Internal method to set gradient function (used by operations)
    ///
    /// Sets the gradient function for this tensor. This is used internally
    /// by tensor operations to record the computation graph for gradtrack.
    ///
    /// # Arguments
    ///
    /// * `grad_fn` - The gradient function to set
    ///
    /// # Implementation Details
    ///
    /// This method is called by tensor operations to register the gradient
    /// computation function. It only sets the gradient function if gradient
    /// tracking is enabled for this tensor.
    pub(crate) fn set_grad_fn(&mut self, grad_fn: GradFn) {
        if self.requires_grad {
            self.grad_fn = grad_fn;
        }
    }

    /// Get a reference to the gradient function (for gradtrack)
    ///
    /// Returns a reference to the gradient function associated with this tensor.
    /// This is used internally by the gradtrack system to compute gradients.
    ///
    /// # Returns
    ///
    /// Reference to the gradient function
    ///
    /// # Implementation Details
    ///
    /// This method is used by the gradtrack engine to access the gradient
    /// computation function during backward pass.
    #[track_caller]
    pub fn grad_fn(&self) -> &GradFn {
        &self.grad_fn
    }

    /// Internal method to set requires_grad (used by gradtrack operations)
    ///
    /// Sets the gradient tracking flag for this tensor. This is used internally
    /// by gradtrack operations to control gradient computation.
    ///
    /// # Arguments
    ///
    /// * `requires_grad` - Whether to enable gradient tracking
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the gradtrack system to control
    /// gradient tracking without triggering additional side effects.
    pub(crate) fn set_requires_grad_internal(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
    }

    /// Internal method to accumulate gradients with optimized operations
    ///
    /// Accumulates a gradient tensor with any existing gradients for this tensor.
    /// This is used internally by the gradtrack system to handle gradient accumulation
    /// during backward pass.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor to accumulate
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the gradtrack engine to accumulate
    /// gradients from multiple backward passes or operations. It only accumulates
    /// gradients if gradient tracking is enabled for this tensor.
    pub(crate) fn accumulate_grad(&mut self, grad: Tensor) {
        if !self.requires_grad {
            return;
        }

        match &self.grad {
            Some(existing_grad) => {
                // Use optimized tensor addition but create new tensor for safety
                let accumulated = existing_grad.add_tensor_optimized(&grad);
                self.grad = Some(Arc::new(accumulated));
            }
            None => {
                self.grad = Some(Arc::new(grad));
            }
        }
    }

    /// Set gradient from external source
    ///
    /// Sets the gradient tensor for this tensor. This is used internally by the
    /// gradtrack system to set gradients during backward pass.
    ///
    /// # Arguments
    ///
    /// * `grad` - The gradient tensor to set
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the gradtrack engine to set gradients
    /// during backward pass. It only sets the gradient if gradient tracking is
    /// enabled for this tensor.
    #[track_caller]
    pub fn set_grad(&mut self, grad: Tensor) {
        if self.requires_grad {
            self.grad = Some(Arc::new(grad));
        }
    }

    /// Clear accumulated gradients for this tensor
    ///
    /// This method is used by optimizers to zero gradients before each backward pass.
    /// It clears any accumulated gradients, allowing for fresh gradient computation.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::ones(vec![2, 3]).with_requires_grad();
    /// tensor.set_grad(Tensor::ones(vec![2, 3]));
    /// assert!(tensor.grad().is_some());
    /// tensor.zero_grad();
    /// assert!(tensor.grad().is_none());
    /// ```
    #[track_caller]
    pub fn zero_grad(&mut self) {
        // Clear locally cached gradient
        self.grad = None;
        // Ensure gradient is also cleared from autograd storage so future backward runs
        // don't observe stale gradients when mixing TLS and shared groups.
        crate::gradtrack::engine::clear_gradient_for_tensor(self.id);
    }

    /// Negate all elements in the tensor in-place
    ///
    /// This is used internally for gradient computation in subtraction operations.
    /// For tensor - tensor operations, the second operand gets a negated gradient.
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the gradtrack system to compute gradients
    /// for subtraction operations. It uses SIMD optimization when available for
    /// better performance.
    #[inline]
    pub(crate) fn negate_inplace(&mut self) {
        if self.shape.size() == 0 {
            return;
        }

        unsafe {
            let ptr = self.data.as_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.negate_simd_avx2(ptr);
                    return;
                }
            }

            // Fallback to scalar operations
            for i in 0..self.shape.size() {
                *ptr.add(i) = -*ptr.add(i);
            }
        }
    }

    /// SIMD-optimized negation using AVX2 instructions
    ///
    /// Performs in-place negation of tensor elements using AVX2 SIMD instructions
    /// for improved performance on x86_64 architectures.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Raw pointer to the tensor data
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `ptr` is a valid pointer to tensor data
    /// - The tensor size matches the actual data size
    /// - The tensor is not moved or dropped during this operation
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by `negate_inplace` when AVX2 is available.
    /// It processes 8 elements per iteration using AVX2 instructions.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn negate_simd_avx2(&self, ptr: *mut f32) {
        use std::arch::x86_64::_mm256_setzero_ps;

        let size = self.shape().size();
        let zero_vec = _mm256_setzero_ps();
        let simd_count = size / 8; // Process 8 elements per iteration
        let mut offset = 0;

        // SIMD loop for negation
        for _ in 0..simd_count {
            use std::arch::x86_64::{_mm256_load_ps, _mm256_store_ps, _mm256_sub_ps};

            let vec = _mm256_load_ps(ptr.add(offset));
            let neg_vec = _mm256_sub_ps(zero_vec, vec);
            _mm256_store_ps(ptr.add(offset), neg_vec);
            offset += 8;
        }

        // Handle remaining elements
        for i in offset..size {
            *ptr.add(i) = -*ptr.add(i);
        }
    }

    // ===== Memory Layout and Optimization API =====

    /// Checks if the tensor data is stored contiguously in memory
    ///
    /// # Returns
    ///
    /// `true` if the tensor data is contiguous, enabling optimized SIMD operations
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// assert!(tensor.is_contiguous());
    /// ```
    #[inline]
    #[track_caller]
    pub fn is_contiguous(&self) -> bool {
        self.shape().is_contiguous()
    }

    /// Gets the memory strides for all dimensions
    ///
    /// # Returns
    ///
    /// Reference to the stride vector for efficient memory access calculations
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// assert_eq!(tensor.strides(), &[12, 4, 1]);
    /// ```
    #[inline]
    #[track_caller]
    pub fn strides(&self) -> &[usize] {
        self.shape.strides()
    }

    /// Gets the memory stride for a specific dimension
    ///
    /// # Arguments
    ///
    /// * `dim` - The dimension index
    ///
    /// # Returns
    ///
    /// The memory stride for the given dimension
    ///
    /// # Panics
    ///
    /// Panics if `dim` is out of bounds
    #[inline]
    #[track_caller]
    pub fn stride(&self, dim: usize) -> usize {
        self.shape.stride(dim)
    }

    /// Calculates the linear memory offset for given multi-dimensional indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Vector of indices for each dimension
    ///
    /// # Returns
    ///
    /// Linear memory offset for direct memory access
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::new(vec![2, 3, 4]);
    /// let offset = tensor.memory_offset(&[1, 2, 3]);
    /// // offset = 1*12 + 2*4 + 3*1 = 23
    /// ```
    #[inline]
    #[track_caller]
    pub fn memory_offset(&self, indices: &[usize]) -> usize {
        self.shape.offset(indices)
    }

    /// Broadcast this tensor with another tensor for element-wise operations
    ///
    /// Returns a tuple containing:
    /// - Broadcasted view of self
    /// - Broadcasted view of other
    /// - Result shape for the operation
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to broadcast with
    ///
    /// # Returns
    ///
    /// A tuple `(broadcasted_self, broadcasted_other, result_shape)`
    #[track_caller]
    pub(crate) fn broadcast_with(
        &self,
        other: &Tensor,
    ) -> Result<
        (Tensor, Tensor, crate::tensor::Shape),
        crate::tensor::ops::broadcasting::BroadcastError,
    > {
        crate::tensor::ops::broadcasting::broadcast_shapes(self, other)
    }

    /// Checks if the tensor data is properly aligned for SIMD operations
    ///
    /// # Returns
    ///
    /// `true` if the tensor data is aligned to 32-byte boundaries for AVX2
    #[inline]
    #[track_caller]
    #[cfg(target_arch = "x86_64")]
    pub(crate) fn is_simd_aligned(&self) -> bool {
        // Maintain AVX2 compatibility for existing SIMD paths
        (self.data.as_ptr() as usize).is_multiple_of(32)
    }

    /// Gets the memory alignment of the tensor data
    ///
    /// # Returns
    ///
    /// The memory alignment in bytes (typically 32 for SIMD optimization)
    #[inline]
    #[track_caller]
    pub fn memory_alignment(&self) -> usize {
        if let Some(owner) = self.allocation_owner() {
            owner.alignment()
        } else {
            // Zero-sized or non-owned views default to conservative 32
            32
        }
    }

    /// Checks if this tensor is broadcastable with another tensor
    ///
    /// # Arguments
    ///
    /// * `other` - The other tensor to check broadcasting compatibility
    ///
    /// # Returns
    ///
    /// `true` if the tensors are broadcastable according to NumPy broadcasting rules
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::new(vec![2, 3, 4]);
    /// let b = Tensor::new(vec![1, 3, 4]);
    /// assert!(a.is_broadcastable_with(&b));
    /// ```
    #[inline]
    #[track_caller]
    pub fn is_broadcastable_with(&self, other: &Tensor) -> bool {
        self.shape.is_broadcastable_with(&other.shape)
    }

    /// Gets the total number of bytes allocated for this tensor
    ///
    /// # Returns
    ///
    /// Total memory footprint in bytes
    #[inline]
    #[track_caller]
    pub fn memory_footprint(&self) -> usize {
        self.shape.size() * std::mem::size_of::<f32>()
    }

    /// Get a single element from the tensor at the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-dimensional indices to access the element
    ///
    /// # Returns
    ///
    /// The value at the specified position
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds or indices length doesn't match tensor rank
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let value = tensor.get(&[0, 1]);
    /// assert_eq!(value, 2.0);
    /// ```
    #[track_caller]
    pub fn get(&self, indices: &[usize]) -> f32 {
        assert_eq!(
            indices.len(),
            self.shape().rank(),
            "Indices length must match tensor rank"
        );

        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < self.shape().dims()[i],
                "Index {} out of bounds for dimension {}",
                idx,
                i
            );
        }

        let offset = self.memory_offset(indices);
        unsafe { *self.as_ptr().add(offset) }
    }

    /// Set a single element in the tensor at the specified indices
    ///
    /// # Arguments
    ///
    /// * `indices` - Multi-dimensional indices to set the element
    /// * `value` - The value to set
    ///
    /// # Panics
    ///
    /// Panics if indices are out of bounds or indices length doesn't match tensor rank
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::new(vec![2, 2]);
    /// tensor.set(&[0, 1], 42.0);
    /// assert_eq!(tensor.get(&[0, 1]), 42.0);
    /// ```
    #[track_caller]
    pub fn set(&mut self, indices: &[usize], value: f32) {
        assert_eq!(
            indices.len(),
            self.shape().rank(),
            "Indices length must match tensor rank"
        );

        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            assert!(
                idx < self.shape().dims()[i],
                "Index {} out of bounds for dimension {}",
                idx,
                i
            );
        }

        let offset = self.memory_offset(indices);
        unsafe {
            *self.as_mut_ptr().add(offset) = value;
        }
    }

    /// Returns a safe slice of the tensor's underlying data
    ///
    /// Provides safe access to the tensor's data without requiring unsafe pointer operations.
    /// This is the preferred way to access tensor data for reading values, comparisons,
    /// and other operations that don't require direct pointer manipulation.
    ///
    /// # Returns
    ///
    /// A slice containing all tensor elements in row-major order
    ///
    /// # Performance
    ///
    /// - **Zero-Cost**: Direct slice creation with no copying
    /// - **Cache-Friendly**: Sequential memory access pattern
    /// - **Safe**: No unsafe code required for basic data access
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let data = tensor.data();
    ///
    /// // Safe indexing and comparisons
    /// assert_eq!(data[0], 1.0);
    /// assert_eq!(data.len(), tensor.size());
    /// ```
    #[inline]
    #[track_caller]
    pub fn data(&self) -> &[f32] {
        if self.size() == 0 {
            return &[];
        }
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.size()) }
    }

    /// Returns a mutable slice of the tensor's underlying data
    ///
    /// Provides safe mutable access to the tensor's data without requiring unsafe
    /// pointer operations. Use this for in-place modifications of tensor values.
    ///
    /// # Returns
    ///
    /// A mutable slice containing all tensor elements in row-major order
    ///
    /// # Performance
    ///
    /// - **Zero-Cost**: Direct slice creation with no copying
    /// - **Cache-Friendly**: Sequential memory access pattern
    /// - **Safe**: No unsafe code required for basic data modification
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let mut tensor = Tensor::new(vec![2, 2]);
    /// let data = tensor.data_mut();
    ///
    /// // Safe indexing for modification
    /// data[0] = 1.0;
    /// data[1] = 2.0;
    ///
    /// assert_eq!(tensor.get(&[0, 0]), 1.0);
    /// assert_eq!(tensor.get(&[0, 1]), 2.0);
    /// ```
    #[inline]
    #[track_caller]
    pub fn data_mut(&mut self) -> &mut [f32] {
        if self.size() == 0 {
            return &mut [];
        }
        // Copy-on-write to protect existing views
        self.ensure_unique_allocation();
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.size()) }
    }

    /// Extract scalar value from single-element tensor
    ///
    /// This method provides a convenient way to extract the scalar value from
    /// tensors that contain exactly one element. This is commonly used with
    /// element iterator results and scalar tensor operations.
    ///
    /// # Returns
    ///
    /// The scalar value contained in this tensor
    ///
    /// # Panics
    ///
    /// Panics if the tensor does not contain exactly one element
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Single-element tensor
    /// let scalar = Tensor::from_slice(&[42.0], vec![1]).unwrap();
    /// assert_eq!(scalar.value(), 42.0);
    /// ```
    #[inline]
    #[track_caller]
    pub fn value(&self) -> f32 {
        assert_eq!(
            self.size(),
            1,
            "value() can only be called on tensors with exactly one element. \
             This tensor has {} elements with shape {:?}",
            self.size(),
            self.shape().dims()
        );
        self.data()[0]
    }

    /// Create a view with a new shape (requires contiguous memory)
    ///
    /// Behaves like PyTorch `view`: tensor must be contiguous and the total
    /// number of elements must remain the same. Supports -1 inference for one dimension.
    ///
    /// # Arguments
    ///
    /// * `new_shape` - New shape for the tensor (can contain -1 for inference)
    ///
    /// # Returns
    ///
    /// A tensor viewing the same data with a new shape
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    /// let y = x.view(vec![2, 2]);
    /// assert_eq!(y.shape().dims(), vec![2, 2]);
    /// ```
    #[track_caller]
    pub fn view(&self, new_shape: Vec<i32>) -> Tensor {
        // PyTorch-like view with single -1 inference; requires contiguity
        let size = self.size();
        let mut infer_idx: Option<usize> = None;
        let mut product: usize = 1;
        let mut dims: Vec<usize> = Vec::with_capacity(new_shape.len());
        for (i, d) in new_shape.iter().enumerate() {
            if *d == -1 {
                assert!(infer_idx.is_none(), "Only one -1 is allowed in view shape");
                infer_idx = Some(i);
                dims.push(1);
            } else {
                assert!(*d > 0, "Negative dims not supported in view shape");
                let du = *d as usize;
                product = product.saturating_mul(du);
                dims.push(du);
            }
        }
        if let Some(pos) = infer_idx {
            assert!(
                product > 0 && size.is_multiple_of(product),
                "View shape incompatible with tensor size"
            );
            dims[pos] = size / product;
        } else {
            assert!(
                product == size,
                "View shape has different number of elements"
            );
        }

        let mut v = match crate::tensor::core::view::reshape_view(self, &dims) {
            Ok(v) => v,
            Err(e) => panic!("view reshape error: {:?}", e),
        };
        if self.requires_grad() && gradtrack::is_grad_enabled() {
            v.set_requires_grad(true);
            let grad_fn = GradFn::Reshape {
                original_shape: self.shape().dims().to_vec(),
            };
            v.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(v.id(), vec![self.id()], grad_fn);
        }
        v
    }

    /// Create an element view for the specified index
    ///
    /// Returns a scalar tensor (shape \[1\]) that views a single element
    /// of the source tensor. Maintains gradient tracking.
    ///
    /// # Arguments
    ///
    /// * `index` - Linear index of the element to view
    ///
    /// # Returns
    ///
    /// A scalar tensor viewing the specified element
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let element = tensor.element_view(1);
    /// assert_eq!(element.value(), 2.0);
    /// ```
    #[track_caller]
    pub fn element_view(&self, index: usize) -> Tensor {
        let mut v = match crate::tensor::core::view::element_view_linear(self, index) {
            Ok(v) => v,
            Err(e) => panic!("element_view error: {:?}", e),
        };
        if self.requires_grad() && gradtrack::is_grad_enabled() {
            v.set_requires_grad(true);
            // Reuse SliceView grad for single-element selection to propagate back to source
            let grad_fn = GradFn::View {
                mapping: crate::gradtrack::grad_fn::ViewMapping::LinearRange {
                    start: index,
                    step: 1,
                    length: 1,
                },
                input_shape: self.shape().dims().to_vec(),
            };
            v.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(v.id(), vec![self.id()], grad_fn);
        }
        v
    }

    /// Create a slice view of the tensor
    ///
    /// Returns a view of a contiguous or strided slice of the source tensor.
    ///
    /// # Arguments
    ///
    /// * `start` - Starting index
    /// * `step` - Step size (1 for contiguous)
    /// * `length` - Number of elements
    ///
    /// # Returns
    ///
    /// A tensor viewing the specified slice
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    /// let slice = tensor.slice_view(1, 2, 2); // [2.0, 4.0]
    /// assert_eq!(slice.get(&[0]), 2.0);
    /// assert_eq!(slice.get(&[1]), 4.0);
    /// ```
    #[track_caller]
    pub fn slice_view(&self, start: usize, step: usize, length: usize) -> Tensor {
        let mut v = match crate::tensor::core::view::slice_view_linear(self, start, step, length) {
            Ok(v) => v,
            Err(e) => panic!("slice_view error: {:?}", e),
        };
        // Ensure correct data() representation for stepped views
        // Offset-only (step==1, start>0) is already contiguous via base pointer
        if step != 1 {
            v = v.contiguous();
        }
        if self.requires_grad() && gradtrack::is_grad_enabled() {
            v.set_requires_grad(true);
            let grad_fn = GradFn::View {
                mapping: crate::gradtrack::grad_fn::ViewMapping::LinearRange {
                    start,
                    step,
                    length,
                },
                input_shape: self.shape().dims().to_vec(),
            };
            v.set_grad_fn(grad_fn.clone());
            GradEngine::register_operation(v.id(), vec![self.id()], grad_fn);
        }
        v
    }

    /// Create a tensor view from raw components
    ///
    /// Creates a tensor that views existing memory with the specified shape and device.
    /// The tensor shares memory with the original allocation through the allocation_owner.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `data` is valid for the number of elements specified by `shape`
    /// - `data` remains valid for the lifetime of the returned tensor
    /// - `allocation_owner` properly manages the memory lifecycle
    ///
    /// # Arguments
    ///
    /// * `data` - Raw pointer to the tensor data
    /// * `shape` - Shape of the tensor view
    /// * `device` - Device where the tensor is located
    /// * `allocation_owner` - Optional shared allocation owner
    ///
    /// # Returns
    ///
    /// A new tensor that views the specified memory
    ///
    /// # Implementation Details
    ///
    /// This method is used internally to create tensor views that share memory
    /// with other tensors. It's primarily used for view operations and memory
    /// management.
    pub(crate) fn from_raw_view(
        data: *const f32,
        shape: Shape,
        device: Device,
        allocation_owner: Option<Arc<Allocation>>,
    ) -> Self {
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        Self {
            data: NonNull::new(data as *mut f32).expect("Data pointer cannot be null"),
            shape,
            device,
            id,
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner,
            graph_group: None,
            _phantom: PhantomData,
        }
    }

    /// Get the allocation owner for this tensor
    ///
    /// Returns the shared allocation owner if this tensor is a view,
    /// or None if this tensor owns its memory directly.
    ///
    /// # Returns
    ///
    /// Optional reference to the allocation owner
    ///
    /// # Implementation Details
    ///
    /// This method is used internally to manage memory lifecycle for tensor views.
    /// It helps determine whether a tensor shares memory with another tensor.
    #[track_caller]
    pub fn allocation_owner(&self) -> Option<&Arc<Allocation>> {
        self.allocation_owner.as_ref()
    }

    /// Create a new tensor with uninitialized memory
    ///
    /// This method allocates memory for a tensor without initializing it to any value.
    /// This is useful for performance-critical operations where the memory will be
    /// immediately overwritten, such as matrix multiplication results.
    ///
    /// # Safety
    ///
    /// The caller must ensure that all memory is written before reading from the tensor.
    /// Reading from uninitialized memory is undefined behavior.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - The dimensions of the tensor
    ///
    /// # Returns
    ///
    /// A tensor with uninitialized memory
    ///
    /// # Performance
    ///
    /// - **Zero Initialization**: Skips memory initialization for maximum performance
    /// - **SIMD Ready**: Properly aligned for vectorized operations
    /// - **Memory Efficient**: Uses optimized alignment strategies
    ///
    /// # Example
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create uninitialized tensor for matmul result
    /// let mut result = Tensor::new_uninitialized(vec![100, 100]);
    /// // Initialize the memory before use
    /// for value in result.data_mut() {
    ///     *value = 0.0;
    /// }
    /// ```
    #[inline]
    #[track_caller]
    pub fn new_uninitialized(shape_dims: Vec<usize>) -> Self {
        let shape = Shape::new(shape_dims);
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if shape.size() == 0 {
            // Handle zero-sized tensors
            return Self {
                data: NonNull::dangling(),
                shape,
                device: current_device(),
                id,
                requires_grad: false,
                retain_grad: false,
                grad: None,
                grad_fn: GradFn::None,
                allocation_owner: None,
                graph_group: None,
                _phantom: PhantomData,
            };
        }

        let (alignment, padded_elems) = compute_allocation_params(shape.size());
        let total_size = padded_elems * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(total_size, alignment)
            .expect("Failed to create layout for tensor data");

        // Allocate memory via shared Allocation (uninitialized)
        let alloc_obj = if use_pool_alloc_enabled() {
            Allocation::new_pooled(padded_elems, alignment, layout)
        } else {
            Allocation::new_uninitialized(padded_elems, alignment, layout)
        };
        let ptr = alloc_obj.ptr;

        debug_assert!(
            alloc_obj.capacity_elems() >= padded_elems,
            "Allocation capacity ({}) smaller than padded elements ({})",
            alloc_obj.capacity_elems(),
            padded_elems
        );

        Self {
            data: ptr,
            shape,
            device: current_device(),
            id,
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner: Some(std::sync::Arc::new(alloc_obj)),
            graph_group: None,
            _phantom: PhantomData,
        }
    }

    /// Create a new uninitialized tensor with an explicit alignment request (in bytes)
    ///
    /// This is intended for internal high-performance paths (e.g., packed GEMM panels)
    /// where stronger alignment such as 64 bytes is desired even on AVX2 systems.
    #[inline]
    #[track_caller]
    pub fn new_uninitialized_aligned(shape_dims: Vec<usize>, alignment_bytes: usize) -> Self {
        let shape = Shape::new(shape_dims);
        let id = TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

        if shape.size() == 0 {
            return Self {
                data: NonNull::dangling(),
                shape,
                device: current_device(),
                id,
                requires_grad: false,
                retain_grad: false,
                grad: None,
                grad_fn: GradFn::None,
                allocation_owner: None,
                graph_group: None,
                _phantom: PhantomData,
            };
        }

        // Preserve the usual element padding policy
        let (_default_align, padded_elems) = compute_allocation_params(shape.size());
        // Honor explicit alignment request (at least 16)
        let alignment = alignment_bytes.max(16);
        let total_size = padded_elems * std::mem::size_of::<f32>();
        let layout = Layout::from_size_align(total_size, alignment)
            .expect("Failed to create layout for tensor data (aligned)");

        let alloc_obj = if use_pool_alloc_enabled() {
            Allocation::new_pooled(padded_elems, alignment, layout)
        } else {
            Allocation::new_uninitialized(padded_elems, alignment, layout)
        };
        let ptr = alloc_obj.ptr;

        debug_assert!(alloc_obj.capacity_elems() >= padded_elems);

        Self {
            data: ptr,
            shape,
            device: current_device(),
            id,
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner: Some(std::sync::Arc::new(alloc_obj)),
            graph_group: None,
            _phantom: PhantomData,
        }
    }
}

// -------------------------------------------------------------------------------------------------
// Fast-path routing helper for iterator and collect APIs
// -------------------------------------------------------------------------------------------------

/// Determine whether to use the no-grad fast path for iterator/collect operations.
///
/// Fast path is selected when either:
/// - No input requires gradients, or
/// - Global grad tracking is disabled for the current thread
#[inline]
pub(crate) fn should_use_fast_path(inputs: &[&Tensor]) -> bool {
    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());
    !any_requires_grad || !crate::gradtrack::is_grad_enabled()
}

#[cfg(test)]
mod memory_alloc_tests {
    use super::*;
    use crate::tensor::core::memory::{
        detect_runtime_simd, simd_alignment_bytes, simd_lane_width_elems, with_no_mem_padding,
        TensorMemoryPool,
    };

    #[test]
    fn test_padding_and_alignment_pool_enabled() {
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let align = simd_alignment_bytes(detect_runtime_simd());

        let req = lane * 3 + 1; // force padding
        let t = Tensor::new(vec![req]);

        assert_eq!(t.capacity_elems() % lane, 0);
        unsafe {
            assert_eq!((t.as_ptr() as usize) % align, 0);
        }
        // Logical size is not a multiple of lane, so aligned SIMD ops are not preferred
        #[cfg(target_arch = "x86_64")]
        assert!(!t.prefer_aligned_simd_ops());

        drop(t);
    }

    #[test]
    fn test_no_padding_with_guard() {
        with_no_mem_padding(|| {
            let lane = simd_lane_width_elems(detect_runtime_simd());
            let req = lane * 3 + 1; // non-multiple
            let t = Tensor::new(vec![req]);

            assert_eq!(t.size(), req);
            #[cfg(target_arch = "x86_64")]
            assert!(!t.prefer_aligned_simd_ops());
        });
    }

    #[test]
    fn test_system_alloc_no_pool_counters_unchanged() {
        let before = TensorMemoryPool::thread_stats();
        let _t1 = Tensor::new(vec![128]);
        let _t2 = Tensor::new(vec![257]);
        let after = TensorMemoryPool::thread_stats();
        assert_eq!(before.allocations, 0);
        assert_eq!(after.allocations, 2);
        assert_eq!(before.deallocations, 0);
    }

    #[test]
    fn test_pool_alloc_and_dealloc_counters_match() {
        let before = TensorMemoryPool::thread_stats();
        {
            let _t1 = Tensor::new(vec![64]);
            let _t2 = Tensor::new(vec![2048]);
            let _t3 = Tensor::new(vec![131072]);
        }
        let after = TensorMemoryPool::thread_stats();
        assert!(after.allocations >= before.allocations + 3);
        assert!(after.deallocations >= before.deallocations + 3);
    }

    #[test]
    fn test_mixed_modes_no_leak_in_pool_stats_scope() {
        let before = TensorMemoryPool::thread_stats();
        {
            let _a = Tensor::new(vec![1000]);
            let _b = Tensor::new(vec![1000]);
            let _c = Tensor::new(vec![2000]);
            let _c = Tensor::new(vec![2000]);
        }

        let after = TensorMemoryPool::thread_stats();
        assert_eq!(
            after.allocations - before.allocations,
            after.deallocations - before.deallocations
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_alignment_helpers_and_hints() {
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let req = lane * 4; // exact multiple
        let t = Tensor::new(vec![req]);
        assert!(t.is_aligned_for_runtime_level());
        assert!(t.prefer_aligned_simd_ops());
        assert!(!t.should_use_unaligned_simd_ops());

        with_no_mem_padding(|| {
            let lane = simd_lane_width_elems(detect_runtime_simd());
            let req = lane * 4 + 1; // not multiple
            let t = Tensor::new(vec![req]);
            assert!(!t.prefer_aligned_simd_ops());
            assert!(t.should_use_unaligned_simd_ops());
        });
    }
}

#[cfg(test)]
mod memory_alloc_additional_tests {
    use super::*;
    use crate::tensor::core::memory::{
        detect_runtime_simd, simd_lane_width_elems, with_no_mem_padding, with_no_mem_pool,
        TensorMemoryPool,
    };
    use std::time::Instant;

    #[test]
    fn test_zero_size_tensors_pool_and_system() {
        let t = Tensor::new(vec![0]);
        assert_eq!(t.size(), 0);
        assert_eq!(t.capacity_elems(), 0);
        assert_eq!(t.data().len(), 0);
        let t = Tensor::new(vec![0]);
        assert_eq!(t.size(), 0);
        assert_eq!(t.capacity_elems(), 0);
        assert_eq!(t.data().len(), 0);
    }

    #[test]
    fn test_capacity_across_various_sizes_padding_modes() {
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let sizes = [
            0,
            1,
            lane - 1,
            lane,
            lane + 1,
            2 * lane - 1,
            2 * lane + 1,
            1000,
            1001,
        ];

        for &n in &sizes {
            let t = Tensor::new(vec![n]);
            let expected_padded = if n == 0 { 0 } else { n.div_ceil(lane) * lane };
            // Pool may round up further than lane padding; ensure at least padded and lane-aligned
            assert!(
                t.capacity_elems() >= expected_padded,
                "n={} lane={} cap={}",
                n,
                lane,
                t.capacity_elems()
            );
            if t.capacity_elems() > 0 {
                assert_eq!(
                    t.capacity_elems() % lane,
                    0,
                    "capacity not lane-multiple: {}",
                    t.capacity_elems()
                );
            }
        }
        with_no_mem_padding(|| {
            for &n in &sizes {
                let t = Tensor::new(vec![n]);
                assert_eq!(t.size(), n);
            }
        });
    }

    #[test]
    fn test_class_boundary_planned_capacity_pool() {
        let boundaries = [
            crate::tensor::core::memory::SMALL_BUFFER_SIZE,
            crate::tensor::core::memory::SMALL_BUFFER_SIZE + 1,
            crate::tensor::core::memory::MEDIUM_BUFFER_SIZE,
            crate::tensor::core::memory::MEDIUM_BUFFER_SIZE + 1,
            crate::tensor::core::memory::LARGE_BUFFER_SIZE,
            crate::tensor::core::memory::LARGE_BUFFER_SIZE + 1,
        ];
        let lane = simd_lane_width_elems(detect_runtime_simd());
        for &n in &boundaries {
            let padded = if n == 0 { 0 } else { n.div_ceil(lane) * lane };
            let planned = TensorMemoryPool::planned_capacity_elems(padded);
            let t = Tensor::new(vec![n]);
            assert_eq!(t.capacity_elems(), planned, "boundary n={}", n);
        }
    }

    #[test]
    fn test_view_does_not_allocate_additional_memory() {
        let before = TensorMemoryPool::thread_stats();
        let t = Tensor::new(vec![256]);
        let _view = t.view(vec![16, 16]);
        let after = TensorMemoryPool::thread_stats();
        // Exactly one allocation happened (for t), view didn't allocate
        assert_eq!(after.allocations, before.allocations + 1);
    }

    #[test]
    fn test_performance_pool_vs_system_small_allocs() {
        let iters = 1000;
        let _ = Tensor::new(vec![64]);
        let _ = Tensor::new(vec![64]);
        let _ = Tensor::new(vec![64]);

        let pool_time = {
            let start = Instant::now();
            for _ in 0..iters {
                let _t = Tensor::new(vec![64]);
            }
            start.elapsed()
        };
        let sys_time = with_no_mem_pool(|| {
            let start = Instant::now();
            for _ in 0..iters {
                let _t = Tensor::new(vec![64]);
            }
            start.elapsed()
        });
        assert!(
            pool_time <= sys_time * 10,
            "pool {:?} vs system {:?}",
            pool_time,
            sys_time
        );
    }

    #[test]
    fn test_performance_padded_vs_unpadded_fill() {
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let n = lane * 2048;

        let padded_time = {
            let t = Tensor::new(vec![n]);
            let mut x = t.clone();
            let start = Instant::now();
            x.fill(1.2345);
            start.elapsed()
        };
        let unpadded_time = with_no_mem_padding(|| {
            let t = Tensor::new(vec![n + 1]);
            let mut x = t.clone();
            let start = Instant::now();
            x.fill(1.2345);
            start.elapsed()
        });
        assert!(
            padded_time <= unpadded_time * 100,
            "padded {:?} vs unpadded {:?}",
            padded_time,
            unpadded_time
        );
    }
}
