use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;

use crate::gradtrack::GradFn;
use crate::tensor::core::TENSOR_ID_COUNTER;
use crate::tensor::Shape;

use super::Tensor;

/// Allocation source tracking for proper deallocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AllocationSource {
    Pool,
    System,
}

/// Shared memory allocation for tensor storage
///
/// Enables zero-copy tensor views by sharing memory allocation between multiple
/// tensors. This struct manages the lifecycle of tensor data and ensures proper
/// cleanup when all views are dropped.
///
/// # Memory Management
///
/// - **Reference Counting**: Uses `Arc` for thread-safe reference counting
/// - **Automatic Cleanup**: Memory freed when last reference is dropped
/// - **Alignment**: Maintains proper alignment for SIMD operations
/// - **Thread Safety**: Safe to share between threads
/// - **RAII Patterns**: Automatic memory management through Drop trait
///
/// # Performance
///
/// - **Zero-Copy Views**: View tensors share memory without copying
/// - **Efficient Cleanup**: Single deallocation when last view drops
/// - **Thread Safe**: Lock-free reference counting
/// - **SIMD Optimized**: Proper alignment for vectorized operations
///
/// # Safety
///
/// The raw pointer is properly managed through RAII patterns and reference
/// counting. Memory is automatically freed when the last tensor view is dropped.
/// All memory access is bounds-checked and properly aligned.
///
/// # Thread Safety
///
/// This type is `Send + Sync` and can be safely shared between threads.
/// All operations are thread-safe through atomic reference counting.
pub struct Allocation {
    /// Raw pointer to the allocated memory
    ///
    /// Points to the beginning of the tensor data. The memory is properly
    /// aligned and sized for the tensor's requirements.
    ///
    /// # Safety
    ///
    /// - Must be valid for `size` elements
    /// - Must be properly aligned for `f32` operations
    /// - Must not be aliased while allocation exists
    pub(super) ptr: NonNull<f32>,

    /// Number of elements in the allocation
    ///
    /// Total number of `f32` elements that can be stored in this allocation.
    /// Used for bounds checking and memory management.
    size: usize,

    /// Memory alignment in bytes
    ///
    /// Alignment requirement for the allocated memory. Used for SIMD operations
    /// and cache optimization.
    alignment: usize,

    /// Source of the allocation for proper deallocation
    source: AllocationSource,
}

// Make Allocation Send + Sync for thread-safe usage with Arc
// Safety: The raw pointer is properly managed through RAII and
// the data is not shared between threads without proper synchronization
unsafe impl Send for Allocation {}
unsafe impl Sync for Allocation {}

impl Allocation {
    /// Creates a new memory allocation for tensor storage
    ///
    /// Allocates memory with the specified size and alignment requirements.
    /// The allocation is optimized for tensor operations and SIMD performance.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of `f32` elements to allocate
    /// * `alignment` - Memory alignment requirement in bytes
    /// * `layout` - Memory layout for allocation
    ///
    /// # Returns
    ///
    /// New allocation with properly aligned memory
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails
    ///
    /// # Safety
    ///
    /// The returned allocation contains valid memory that must be freed
    /// when the allocation is dropped.
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the tensor system to allocate memory
    /// for tensor data. The allocation is properly aligned for SIMD operations
    /// and managed through RAII patterns for automatic cleanup.
    pub(super) fn new(size: usize, alignment: usize, layout: Layout) -> Self {
        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                panic!("Failed to allocate memory for tensor");
            }
            NonNull::new_unchecked(p as *mut f32)
        };
        // Ensure alignment guarantees are met
        let addr = ptr.as_ptr() as usize;
        assert_eq!(
            addr % alignment,
            0,
            "System allocation not aligned to {} bytes",
            alignment
        );
        Allocation {
            ptr,
            size,
            alignment,
            source: AllocationSource::System,
        }
    }

    /// Creates a new memory allocation for tensor storage (uninitialized)
    ///
    /// Allocates memory with the specified size and alignment requirements.
    /// The allocation is optimized for tensor operations and SIMD performance.
    /// This method is identical to `new` but emphasizes that the memory is
    /// uninitialized and should be properly initialized before use.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of `f32` elements to allocate
    /// * `alignment` - Memory alignment requirement in bytes
    /// * `layout` - Memory layout for allocation
    ///
    /// # Returns
    ///
    /// New allocation with properly aligned memory
    ///
    /// # Panics
    ///
    /// Panics if memory allocation fails
    ///
    /// # Safety
    ///
    /// The returned allocation contains valid but uninitialized memory that must
    /// be properly initialized before use and freed when the allocation is dropped.
    ///
    /// # Implementation Details
    ///
    /// This method is used internally by the tensor system to allocate uninitialized
    /// memory for tensor data. The memory must be properly initialized before use
    /// to avoid undefined behavior.
    pub(crate) fn new_uninitialized(size: usize, alignment: usize, layout: Layout) -> Self {
        let ptr = unsafe {
            let p = alloc(layout);
            if p.is_null() {
                panic!("Failed to allocate memory for tensor");
            }
            NonNull::new_unchecked(p as *mut f32)
        };
        let addr = ptr.as_ptr() as usize;
        assert_eq!(
            addr % alignment,
            0,
            "System allocation not aligned to {} bytes",
            alignment
        );
        Allocation {
            ptr,
            size,
            alignment,
            source: AllocationSource::System,
        }
    }

    /// Creates a new memory allocation using the thread-local memory pool with fallback.
    pub(crate) fn new_pooled(size: usize, alignment: usize, layout: Layout) -> Self {
        let planned = crate::tensor::core::memory::TensorMemoryPool::planned_capacity_elems(size);
        match crate::tensor::core::memory::TensorMemoryPool::allocate(size, alignment) {
            Some(ptr) => {
                let addr = ptr.as_ptr() as usize;
                let effective_alignment = alignment.max(std::mem::align_of::<f32>());

                if !addr.is_multiple_of(effective_alignment) {
                    return Self::new(size, alignment, layout);
                }
                Allocation {
                    ptr,
                    size: planned,
                    alignment,
                    source: AllocationSource::Pool,
                }
            }
            None => Self::new(size, alignment, layout),
        }
    }

    // Pool-based allocation removed for simplicity and to avoid cross-thread leaks

    /// Returns the memory alignment of this allocation
    ///
    /// The alignment is determined by the tensor size and optimized for
    /// SIMD operations (SSE, AVX2) and cache performance.
    pub fn alignment(&self) -> usize {
        self.alignment
    }

    /// Returns the total capacity of this allocation in number of f32 elements.
    /// This may be greater than the logical tensor length due to SIMD padding.
    #[inline]
    pub fn capacity_elems(&self) -> usize {
        self.size
    }
}

impl Drop for Allocation {
    /// Frees the allocated memory when the allocation is dropped
    ///
    /// Ensures proper cleanup of tensor memory to prevent memory leaks.
    /// Only deallocates if the allocation contains valid memory (size > 0).
    ///
    /// # Safety
    ///
    /// This function is safe because it only deallocates memory that was
    /// properly allocated by the `new` method and maintains the same layout.
    fn drop(&mut self) {
        if self.size == 0 {
            return;
        }
        match self.source {
            AllocationSource::Pool => {
                // Attempt to return to the current thread's pool. If TLS is unavailable
                // or the pointer isn't known to this pool (e.g., dropped on a different
                // thread than it was allocated), DO NOT deallocate here to avoid double free.
                // The owning pool retains the allocation and will reclaim it or free it when dropped.
                let _ = crate::tensor::core::memory::TensorMemoryPool::try_deallocate(self.ptr);
            }
            AllocationSource::System => unsafe {
                let layout =
                    Layout::from_size_align(self.size * std::mem::size_of::<f32>(), self.alignment)
                        .expect("Invalid layout for allocation deallocation");
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            },
        }
    }
}

impl Tensor {
    /// Creates a zero-copy view tensor with provided shape sharing memory
    ///
    /// Creates a new tensor that shares the same memory allocation as this tensor
    /// but with different shape and stride information. This enables efficient
    /// tensor transformations without copying data.
    pub(crate) fn create_view_with_shape(&self, new_shape: Shape) -> Tensor {
        let mut t = Tensor {
            data: self.data,
            shape: new_shape,
            device: self.device,
            id: TENSOR_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            requires_grad: false,
            retain_grad: false,
            grad: None,
            grad_fn: GradFn::None,
            allocation_owner: self.allocation_owner.clone(),
            graph_group: self.graph_group.clone(),
            _phantom: PhantomData,
        };
        if self.requires_grad {
            t.requires_grad = true;
        }
        t
    }
}
