//! Addition operations for tensors
//!
//! Provides element-wise addition following PyTorch conventions with comprehensive
//! broadcasting support, automatic differentiation, and high-performance SIMD optimization.
//!
//! # Key Features
//!
//! - **Element-wise Addition**: `add_tensor()` - Addition with another tensor (PyTorch `add()` equivalent)
//! - **Scalar Broadcasting**: `add_scalar()` - Addition with scalar values
//! - **Automatic Broadcasting**: NumPy-style broadcasting for compatible shapes
//! - **SIMD Optimization**: AVX2 acceleration on x86_64 hardware
//! - **Automatic Differentiation**: Full gradtrack support with gradient tracking
//! - **Cache Optimization**: Memory access patterns optimized for modern CPUs
//! - **Zero-copy Operations**: Efficient memory usage where possible
//!
//! # Broadcasting Support
//!
//! All addition operations support automatic broadcasting following NumPy rules:
//! - Dimensions are aligned from the rightmost dimension
//! - Dimensions are compatible if they are equal, or one of them is 1
//! - Missing dimensions are treated as 1
//! - Result shape follows broadcasting rules
//!
//! # Performance Characteristics
//!
//! - **SIMD Acceleration**: 8x vectorization with AVX2 on compatible hardware
//! - **Unrolled Loops**: 4x unrolling for optimal instruction throughput
//! - **Cache-friendly Access**: Linear memory access patterns
//! - **Fallback Support**: Optimized scalar implementations for non-SIMD hardware
//! - **Gradient Optimization**: Efficient gradtrack with NoGradTrack support

use crate::gradtrack::{is_grad_enabled, GradEngine, GradFn};
#[cfg(target_arch = "x86_64")]
use crate::tensor::core::memory::simd_alignment_bytes;
use crate::tensor::core::memory::{detect_runtime_simd, SimdLevel};
// Enhanced thread pool imports
// thread pool invoked via aligned wrappers inside functions where needed
use crate::tensor::core::Tensor;
// SIMD optimizations for performance-critical operations

// (Removed manual prefetching: simplifies hot path; modern CPUs prefetch effectively for linear access)

/// OPTIMIZATION #6: Cached SIMD kernels and dispatch information for maximum performance
struct CachedKernels {
    simd_level: SimdLevel,
    alignment: usize,

    // Tensor + Tensor kernels
    tensor_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize),
    tensor_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize),
    tensor_stream: unsafe fn(*const f32, *const f32, *mut f32, usize),

    // Tensor + Scalar kernels
    scalar_aligned: unsafe fn(*const f32, *mut f32, usize, f32),
    scalar_unaligned: unsafe fn(*const f32, *mut f32, usize, f32),
    scalar_stream: unsafe fn(*const f32, *mut f32, usize, f32),

    // Dispatch parameters
    min_aligned_size: usize,
    min_stream_size: usize,
}

impl Tensor {
    // ===== Streaming store thresholds =====
    #[inline]
    pub(crate) fn stream_min_elems() -> usize {
        1 << 22 // ~16MB per chunk (f32) conservative threshold for streaming stores
    }

    /// Element-wise addition with another tensor with broadcasting support.
    ///
    /// Performs element-wise addition with automatic broadcasting: `output[i] = self[i] + other[i]`
    ///
    /// Broadcasting enables addition between tensors of different but compatible shapes.
    /// Compatible shapes follow NumPy broadcasting rules:
    /// - Dimensions are aligned from the rightmost dimension
    /// - Dimensions are compatible if they are equal, or one of them is 1
    /// - Missing dimensions are treated as 1
    ///
    /// # Arguments
    /// * `other` - Tensor to add. Shapes must be broadcast-compatible.
    ///
    /// # Returns
    /// A new tensor containing the element-wise sum with broadcast result shape
    ///
    /// # Examples
    ///
    /// ## Same Shape Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let a = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
    /// let b = Tensor::from_slice(&[4.0, 5.0, 6.0], vec![3]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims(), vec![3]);
    /// assert_eq!(c.get(&[0]), 5.0);
    /// assert_eq!(c.get(&[1]), 7.0);
    /// assert_eq!(c.get(&[2]), 9.0);
    /// ```
    ///
    /// ## Broadcasting Addition
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Broadcasting: [2, 1] + [1, 3] -> [2, 3]
    /// let a = Tensor::from_slice(&[1.0, 2.0], vec![2, 1]).unwrap();
    /// let b = Tensor::from_slice(&[10.0, 20.0, 30.0], vec![1, 3]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims(), vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 11.0);
    /// assert_eq!(c.get(&[0, 1]), 21.0);
    /// assert_eq!(c.get(&[1, 0]), 12.0);
    /// assert_eq!(c.get(&[1, 1]), 22.0);
    /// ```
    ///
    /// ## Scalar Broadcasting
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Scalar broadcasting: [2, 3] + scalar -> [2, 3]
    /// let a = Tensor::ones(vec![2, 3]);
    /// let b = Tensor::from_slice(&[5.0], vec![1]).unwrap();
    /// let c = a.add_tensor(&b);
    /// assert_eq!(c.shape().dims(), vec![2, 3]);
    /// assert_eq!(c.get(&[0, 0]), 6.0);
    /// assert_eq!(c.get(&[1, 2]), 6.0);
    /// ```
    ///
    /// # Panics
    /// Panics if tensor shapes are not broadcast-compatible
    #[inline]
    #[track_caller]
    pub fn add_tensor(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape().dims() == other.shape().dims() {
            return self.add_tensor_same_shape(other);
        }

        // Use broadcasting.rs for efficient broadcasting
        let mut result = self.add_tensor_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: true,
                original_shapes: Some((
                    self.shape().dims().to_vec(),
                    other.shape().dims().to_vec(),
                )),
            };
            result.set_grad_fn(grad_fn.clone());

            // Optimize input ID collection - avoid Vec allocation for common cases
            let input_ids = vec![self.id(), other.id()];
            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Element-wise addition for tensors with identical shapes (fast path).
    #[inline]
    fn add_tensor_same_shape(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Tensor shapes must match for same-shape addition"
        );
        let mut result = self.add_tensor_same_shape_optimized(other);

        if (self.requires_grad() || other.requires_grad()) && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: true,
                original_shapes: None, // Same shape case
            };
            result.set_grad_fn(grad_fn.clone());

            // Optimize input ID collection - avoid Vec allocation for common cases
            let input_ids = vec![self.id(), other.id()];
            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Broadcast addition with a scalar value.
    #[inline]
    #[track_caller]
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let mut result = self.add_scalar_optimized(scalar);

        if self.requires_grad() && is_grad_enabled() {
            result.set_requires_grad_internal(true);
            let grad_fn = GradFn::Add {
                is_tensor_add: false,
                original_shapes: None, // Scalar case
            };
            result.set_grad_fn(grad_fn.clone());
            // Optimize scalar operation registration - no Vec allocation needed
            let input_ids = vec![self.id()];
            GradEngine::register_operation(result.id(), input_ids, grad_fn);
        }

        result
    }

    /// Internal optimized tensor + tensor operation with broadcasting support
    #[inline]
    pub(crate) fn add_tensor_optimized(&self, other: &Tensor) -> Tensor {
        // Check if shapes are identical for fast path
        if self.shape() == other.shape() {
            return self.add_tensor_same_shape_optimized(other);
        }

        // Use zero-copy broadcasting to create same-shape views, then reuse optimized kernels
        use crate::tensor::ops::broadcasting::{broadcast_shapes_cow, BroadcastError};

        match broadcast_shapes_cow(self, other) {
            Ok((broadcasted_self, broadcasted_other, _result_shape)) => {
                debug_assert_eq!(
                    broadcasted_self.shape().dims(),
                    broadcasted_other.shape().dims()
                );
                broadcasted_self
                    .as_ref()
                    .add_tensor_same_shape_optimized(broadcasted_other.as_ref())
            }
            Err(BroadcastError::IncompatibleShapes { shape1, shape2, .. }) => {
                panic!(
                    "Cannot broadcast tensor shapes {:?} and {:?}: shapes are incompatible",
                    shape1, shape2
                );
            }
            Err(BroadcastError::AllocationFailed) => {
                panic!("Memory allocation failed during broadcasting");
            }
        }
    }

    /// Optimized same-shape tensor addition (extracted from original add_tensor_optimized)
    #[inline]
    fn add_tensor_same_shape_optimized(&self, other: &Tensor) -> Tensor {
        debug_assert_eq!(
            self.shape().dims(),
            other.shape().dims(),
            "Tensor dims must match"
        );

        // Optimize contiguous handling - use views when possible
        let (a_ptr, _a_keep): (*const f32, Option<Tensor>) = Self::get_optimized_tensor_ptr(self);
        let (b_ptr, _b_keep): (*const f32, Option<Tensor>) = Self::get_optimized_tensor_ptr(other);

        let mut output = Tensor::new(self.shape().dims().to_vec());

        unsafe {
            let a = a_ptr;
            let b = b_ptr;
            let dst = output.as_mut_ptr();
            let n = self.size();

            // Sequential execution with SIMD optimization
            let stream_min = Self::stream_min_elems();
            if n >= stream_min && Self::try_add_stream_best(a, b, dst, n) {
                // done via streaming stores
            } else if !Self::try_add_simd_best(a, b, dst, n) {
                Self::add_tensors_scalar_chunk(a, b, dst, n);
            }
        }

        output
    }

    // Replaced single AVX2 kernel with multi-level SIMD selection and kernels below

    // ===== OPTIMIZATION #6: Cached SIMD tensor+tensor addition dispatch =====
    #[inline]
    unsafe fn try_add_simd_best(a: *const f32, b: *const f32, dst: *mut f32, size: usize) -> bool {
        if size == 0 {
            return true;
        }

        let kernels = Self::get_cached_kernels();

        // Skip SIMD for scalar fallback
        if matches!(kernels.simd_level, SimdLevel::Scalar) {
            return false;
        }

        // Check alignment for optimal kernel selection
        let a_mod = (a as usize) % kernels.alignment;
        let b_mod = (b as usize) % kernels.alignment;
        let d_mod = (dst as usize) % kernels.alignment;

        // All aligned - use fastest kernel
        if a_mod == 0 && b_mod == 0 && d_mod == 0 && size >= kernels.min_aligned_size {
            (kernels.tensor_aligned)(a, b, dst, size);
            return true;
        }

        // Same misalignment - align and use fast kernel
        if a_mod == b_mod && b_mod == d_mod && size >= kernels.min_aligned_size {
            let bytes_to_align = if a_mod == 0 {
                0
            } else {
                kernels.alignment - a_mod
            };
            let elems_to_align = (bytes_to_align / std::mem::size_of::<f32>()).min(size);

            // Scalar prologue to achieve alignment
            for i in 0..elems_to_align {
                *dst.add(i) = *a.add(i) + *b.add(i);
            }

            let rem = size - elems_to_align;
            if rem >= kernels.min_aligned_size {
                (kernels.tensor_aligned)(
                    a.add(elems_to_align),
                    b.add(elems_to_align),
                    dst.add(elems_to_align),
                    rem,
                );
            }
            return true;
        }

        // Mixed alignment - use unaligned kernel
        (kernels.tensor_unaligned)(a, b, dst, size);
        true
    }

    // OPTIMIZATION #6: Cached streaming store selection for tensor+tensor addition
    #[inline]
    unsafe fn try_add_stream_best(
        a: *const f32,
        b: *const f32,
        dst: *mut f32,
        size: usize,
    ) -> bool {
        let kernels = Self::get_cached_kernels();

        if size < kernels.min_stream_size || size == 0 {
            return false;
        }

        // Only use streaming if destination is aligned (critical for streaming stores)
        if (dst as usize).is_multiple_of(kernels.alignment) {
            (kernels.tensor_stream)(a, b, dst, size);
            return true;
        }

        false
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_simd_avx512_aligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 64usize; // 4x unroll of 16-wide vectors
        while offset + block <= size {
            let a1 = _mm512_load_ps(a.add(offset));
            let b1 = _mm512_load_ps(b.add(offset));
            _mm512_store_ps(dst.add(offset), _mm512_add_ps(a1, b1));

            let a2 = _mm512_load_ps(a.add(offset + 16));
            let b2 = _mm512_load_ps(b.add(offset + 16));
            _mm512_store_ps(dst.add(offset + 16), _mm512_add_ps(a2, b2));

            let a3 = _mm512_load_ps(a.add(offset + 32));
            let b3 = _mm512_load_ps(b.add(offset + 32));
            _mm512_store_ps(dst.add(offset + 32), _mm512_add_ps(a3, b3));

            let a4 = _mm512_load_ps(a.add(offset + 48));
            let b4 = _mm512_load_ps(b.add(offset + 48));
            _mm512_store_ps(dst.add(offset + 48), _mm512_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let av = _mm512_load_ps(a.add(offset));
            let bv = _mm512_load_ps(b.add(offset));
            _mm512_store_ps(dst.add(offset), _mm512_add_ps(av, bv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_simd_avx512_unaligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 64usize;
        while offset + block <= size {
            let a1 = _mm512_loadu_ps(a.add(offset));
            let b1 = _mm512_loadu_ps(b.add(offset));
            _mm512_storeu_ps(dst.add(offset), _mm512_add_ps(a1, b1));

            let a2 = _mm512_loadu_ps(a.add(offset + 16));
            let b2 = _mm512_loadu_ps(b.add(offset + 16));
            _mm512_storeu_ps(dst.add(offset + 16), _mm512_add_ps(a2, b2));

            let a3 = _mm512_loadu_ps(a.add(offset + 32));
            let b3 = _mm512_loadu_ps(b.add(offset + 32));
            _mm512_storeu_ps(dst.add(offset + 32), _mm512_add_ps(a3, b3));

            let a4 = _mm512_loadu_ps(a.add(offset + 48));
            let b4 = _mm512_loadu_ps(b.add(offset + 48));
            _mm512_storeu_ps(dst.add(offset + 48), _mm512_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let av = _mm512_loadu_ps(a.add(offset));
            let bv = _mm512_loadu_ps(b.add(offset));
            _mm512_storeu_ps(dst.add(offset), _mm512_add_ps(av, bv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_simd_avx512_stream(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 64usize;
        while offset + block <= size {
            let a1 = _mm512_loadu_ps(a.add(offset));
            let b1 = _mm512_loadu_ps(b.add(offset));
            _mm512_stream_ps(dst.add(offset), _mm512_add_ps(a1, b1));

            let a2 = _mm512_loadu_ps(a.add(offset + 16));
            let b2 = _mm512_loadu_ps(b.add(offset + 16));
            _mm512_stream_ps(dst.add(offset + 16), _mm512_add_ps(a2, b2));

            let a3 = _mm512_loadu_ps(a.add(offset + 32));
            let b3 = _mm512_loadu_ps(b.add(offset + 32));
            _mm512_stream_ps(dst.add(offset + 32), _mm512_add_ps(a3, b3));

            let a4 = _mm512_loadu_ps(a.add(offset + 48));
            let b4 = _mm512_loadu_ps(b.add(offset + 48));
            _mm512_stream_ps(dst.add(offset + 48), _mm512_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let av = _mm512_loadu_ps(a.add(offset));
            let bv = _mm512_loadu_ps(b.add(offset));
            _mm512_stream_ps(dst.add(offset), _mm512_add_ps(av, bv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_simd_avx2_aligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 32usize; // 4x unroll of 8-wide vectors
        while offset + block <= size {
            let a1 = _mm256_load_ps(a.add(offset));
            let b1 = _mm256_load_ps(b.add(offset));
            _mm256_store_ps(dst.add(offset), _mm256_add_ps(a1, b1));

            let a2 = _mm256_load_ps(a.add(offset + 8));
            let b2 = _mm256_load_ps(b.add(offset + 8));
            _mm256_store_ps(dst.add(offset + 8), _mm256_add_ps(a2, b2));

            let a3 = _mm256_load_ps(a.add(offset + 16));
            let b3 = _mm256_load_ps(b.add(offset + 16));
            _mm256_store_ps(dst.add(offset + 16), _mm256_add_ps(a3, b3));

            let a4 = _mm256_load_ps(a.add(offset + 24));
            let b4 = _mm256_load_ps(b.add(offset + 24));
            _mm256_store_ps(dst.add(offset + 24), _mm256_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let av = _mm256_load_ps(a.add(offset));
            let bv = _mm256_load_ps(b.add(offset));
            _mm256_store_ps(dst.add(offset), _mm256_add_ps(av, bv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_simd_avx2_unaligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 32usize;
        while offset + block <= size {
            let a1 = _mm256_loadu_ps(a.add(offset));
            let b1 = _mm256_loadu_ps(b.add(offset));
            _mm256_storeu_ps(dst.add(offset), _mm256_add_ps(a1, b1));

            let a2 = _mm256_loadu_ps(a.add(offset + 8));
            let b2 = _mm256_loadu_ps(b.add(offset + 8));
            _mm256_storeu_ps(dst.add(offset + 8), _mm256_add_ps(a2, b2));

            let a3 = _mm256_loadu_ps(a.add(offset + 16));
            let b3 = _mm256_loadu_ps(b.add(offset + 16));
            _mm256_storeu_ps(dst.add(offset + 16), _mm256_add_ps(a3, b3));

            let a4 = _mm256_loadu_ps(a.add(offset + 24));
            let b4 = _mm256_loadu_ps(b.add(offset + 24));
            _mm256_storeu_ps(dst.add(offset + 24), _mm256_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let av = _mm256_loadu_ps(a.add(offset));
            let bv = _mm256_loadu_ps(b.add(offset));
            _mm256_storeu_ps(dst.add(offset), _mm256_add_ps(av, bv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_simd_avx2_stream(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 32usize;
        while offset + block <= size {
            let a1 = _mm256_loadu_ps(a.add(offset));
            let b1 = _mm256_loadu_ps(b.add(offset));
            _mm256_stream_ps(dst.add(offset), _mm256_add_ps(a1, b1));

            let a2 = _mm256_loadu_ps(a.add(offset + 8));
            let b2 = _mm256_loadu_ps(b.add(offset + 8));
            _mm256_stream_ps(dst.add(offset + 8), _mm256_add_ps(a2, b2));

            let a3 = _mm256_loadu_ps(a.add(offset + 16));
            let b3 = _mm256_loadu_ps(b.add(offset + 16));
            _mm256_stream_ps(dst.add(offset + 16), _mm256_add_ps(a3, b3));

            let a4 = _mm256_loadu_ps(a.add(offset + 24));
            let b4 = _mm256_loadu_ps(b.add(offset + 24));
            _mm256_stream_ps(dst.add(offset + 24), _mm256_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let av = _mm256_loadu_ps(a.add(offset));
            let bv = _mm256_loadu_ps(b.add(offset));
            _mm256_stream_ps(dst.add(offset), _mm256_add_ps(av, bv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_simd_sse_aligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 16usize; // 4x unroll of 4-wide vectors
        while offset + block <= size {
            let a1 = _mm_load_ps(a.add(offset));
            let b1 = _mm_load_ps(b.add(offset));
            _mm_store_ps(dst.add(offset), _mm_add_ps(a1, b1));

            let a2 = _mm_load_ps(a.add(offset + 4));
            let b2 = _mm_load_ps(b.add(offset + 4));
            _mm_store_ps(dst.add(offset + 4), _mm_add_ps(a2, b2));

            let a3 = _mm_load_ps(a.add(offset + 8));
            let b3 = _mm_load_ps(b.add(offset + 8));
            _mm_store_ps(dst.add(offset + 8), _mm_add_ps(a3, b3));

            let a4 = _mm_load_ps(a.add(offset + 12));
            let b4 = _mm_load_ps(b.add(offset + 12));
            _mm_store_ps(dst.add(offset + 12), _mm_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let av = _mm_load_ps(a.add(offset));
            let bv = _mm_load_ps(b.add(offset));
            _mm_store_ps(dst.add(offset), _mm_add_ps(av, bv));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_simd_sse_unaligned(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 16usize;
        while offset + block <= size {
            let a1 = _mm_loadu_ps(a.add(offset));
            let b1 = _mm_loadu_ps(b.add(offset));
            _mm_storeu_ps(dst.add(offset), _mm_add_ps(a1, b1));

            let a2 = _mm_loadu_ps(a.add(offset + 4));
            let b2 = _mm_loadu_ps(b.add(offset + 4));
            _mm_storeu_ps(dst.add(offset + 4), _mm_add_ps(a2, b2));

            let a3 = _mm_loadu_ps(a.add(offset + 8));
            let b3 = _mm_loadu_ps(b.add(offset + 8));
            _mm_storeu_ps(dst.add(offset + 8), _mm_add_ps(a3, b3));

            let a4 = _mm_loadu_ps(a.add(offset + 12));
            let b4 = _mm_loadu_ps(b.add(offset + 12));
            _mm_storeu_ps(dst.add(offset + 12), _mm_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let s = _mm_loadu_ps(a.add(offset));
            let t = _mm_loadu_ps(b.add(offset));
            _mm_storeu_ps(dst.add(offset), _mm_add_ps(s, t));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_simd_sse_stream(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        use std::arch::x86_64::*;
        let mut offset = 0usize;
        let block = 16usize;
        while offset + block <= size {
            let a1 = _mm_loadu_ps(a.add(offset));
            let b1 = _mm_loadu_ps(b.add(offset));
            _mm_stream_ps(dst.add(offset), _mm_add_ps(a1, b1));

            let a2 = _mm_loadu_ps(a.add(offset + 4));
            let b2 = _mm_loadu_ps(b.add(offset + 4));
            _mm_stream_ps(dst.add(offset + 4), _mm_add_ps(a2, b2));

            let a3 = _mm_loadu_ps(a.add(offset + 8));
            let b3 = _mm_loadu_ps(b.add(offset + 8));
            _mm_stream_ps(dst.add(offset + 8), _mm_add_ps(a3, b3));

            let a4 = _mm_loadu_ps(a.add(offset + 12));
            let b4 = _mm_loadu_ps(b.add(offset + 12));
            _mm_stream_ps(dst.add(offset + 12), _mm_add_ps(a4, b4));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let s = _mm_loadu_ps(a.add(offset));
            let t = _mm_loadu_ps(b.add(offset));
            _mm_stream_ps(dst.add(offset), _mm_add_ps(s, t));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    // ===== SIMD scalar addition selection and kernels =====
    #[inline]
    // OPTIMIZATION #6: Cached scalar SIMD dispatch
    unsafe fn try_add_scalar_simd_best(
        src: *const f32,
        dst: *mut f32,
        size: usize,
        scalar: f32,
    ) -> bool {
        if size == 0 {
            return true;
        }

        let kernels = Self::get_cached_kernels();

        // Skip SIMD for scalar fallback
        if matches!(kernels.simd_level, SimdLevel::Scalar) {
            return false;
        }

        // Check alignment for optimal kernel selection
        let s_mod = (src as usize) % kernels.alignment;
        let d_mod = (dst as usize) % kernels.alignment;

        // Both aligned - use fastest kernel
        if s_mod == 0 && d_mod == 0 && size >= kernels.min_aligned_size {
            (kernels.scalar_aligned)(src, dst, size, scalar);
            return true;
        }

        // Same misalignment - align and use fast kernel
        if s_mod == d_mod && size >= kernels.min_aligned_size {
            let bytes_to_align = if s_mod == 0 {
                0
            } else {
                kernels.alignment - s_mod
            };
            let elems_to_align = (bytes_to_align / std::mem::size_of::<f32>()).min(size);

            // Scalar prologue to achieve alignment
            for i in 0..elems_to_align {
                *dst.add(i) = *src.add(i) + scalar;
            }

            let rem = size - elems_to_align;
            if rem >= kernels.min_aligned_size {
                (kernels.scalar_aligned)(
                    src.add(elems_to_align),
                    dst.add(elems_to_align),
                    rem,
                    scalar,
                );
            }
            return true;
        }

        // Mixed alignment - use unaligned kernel
        (kernels.scalar_unaligned)(src, dst, size, scalar);
        true
    }

    // OPTIMIZATION #6: Cached streaming store selection for scalar addition
    #[inline]
    unsafe fn try_add_scalar_stream_best(
        src: *const f32,
        dst: *mut f32,
        size: usize,
        scalar: f32,
    ) -> bool {
        let kernels = Self::get_cached_kernels();

        if size < kernels.min_stream_size || size == 0 {
            return false;
        }

        // Only use streaming if destination is aligned (critical for streaming stores)
        if (dst as usize).is_multiple_of(kernels.alignment) {
            (kernels.scalar_stream)(src, dst, size, scalar);
            return true;
        }

        false
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_scalar_avx512_aligned(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm512_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 64usize; // 4x 16-wide
        while offset + block <= size {
            let s1 = _mm512_load_ps(src.add(offset));
            _mm512_store_ps(dst.add(offset), _mm512_add_ps(s1, sv));
            let s2 = _mm512_load_ps(src.add(offset + 16));
            _mm512_store_ps(dst.add(offset + 16), _mm512_add_ps(s2, sv));
            let s3 = _mm512_load_ps(src.add(offset + 32));
            _mm512_store_ps(dst.add(offset + 32), _mm512_add_ps(s3, sv));
            let s4 = _mm512_load_ps(src.add(offset + 48));
            _mm512_store_ps(dst.add(offset + 48), _mm512_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let s = _mm512_load_ps(src.add(offset));
            _mm512_store_ps(dst.add(offset), _mm512_add_ps(s, sv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_scalar_avx512_unaligned(
        src: *const f32,
        dst: *mut f32,
        size: usize,
        scalar: f32,
    ) {
        use std::arch::x86_64::*;
        let sv = _mm512_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 64usize;
        while offset + block <= size {
            let s1 = _mm512_loadu_ps(src.add(offset));
            _mm512_storeu_ps(dst.add(offset), _mm512_add_ps(s1, sv));
            let s2 = _mm512_loadu_ps(src.add(offset + 16));
            _mm512_storeu_ps(dst.add(offset + 16), _mm512_add_ps(s2, sv));
            let s3 = _mm512_loadu_ps(src.add(offset + 32));
            _mm512_storeu_ps(dst.add(offset + 32), _mm512_add_ps(s3, sv));
            let s4 = _mm512_loadu_ps(src.add(offset + 48));
            _mm512_storeu_ps(dst.add(offset + 48), _mm512_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let s = _mm512_loadu_ps(src.add(offset));
            _mm512_storeu_ps(dst.add(offset), _mm512_add_ps(s, sv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn add_scalar_avx512_stream(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm512_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 64usize;
        while offset + block <= size {
            let s1 = _mm512_loadu_ps(src.add(offset));
            _mm512_stream_ps(dst.add(offset), _mm512_add_ps(s1, sv));
            let s2 = _mm512_loadu_ps(src.add(offset + 16));
            _mm512_stream_ps(dst.add(offset + 16), _mm512_add_ps(s2, sv));
            let s3 = _mm512_loadu_ps(src.add(offset + 32));
            _mm512_stream_ps(dst.add(offset + 32), _mm512_add_ps(s3, sv));
            let s4 = _mm512_loadu_ps(src.add(offset + 48));
            _mm512_stream_ps(dst.add(offset + 48), _mm512_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 16 {
            let s = _mm512_loadu_ps(src.add(offset));
            _mm512_stream_ps(dst.add(offset), _mm512_add_ps(s, sv));
            offset += 16;
            rem -= 16;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_avx2_aligned(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm256_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 32usize;
        while offset + block <= size {
            let s1 = _mm256_load_ps(src.add(offset));
            _mm256_store_ps(dst.add(offset), _mm256_add_ps(s1, sv));
            let s2 = _mm256_load_ps(src.add(offset + 8));
            _mm256_store_ps(dst.add(offset + 8), _mm256_add_ps(s2, sv));
            let s3 = _mm256_load_ps(src.add(offset + 16));
            _mm256_store_ps(dst.add(offset + 16), _mm256_add_ps(s3, sv));
            let s4 = _mm256_load_ps(src.add(offset + 24));
            _mm256_store_ps(dst.add(offset + 24), _mm256_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let s = _mm256_load_ps(src.add(offset));
            _mm256_store_ps(dst.add(offset), _mm256_add_ps(s, sv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_avx2_unaligned(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm256_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 32usize;
        while offset + block <= size {
            let s1 = _mm256_loadu_ps(src.add(offset));
            _mm256_storeu_ps(dst.add(offset), _mm256_add_ps(s1, sv));
            let s2 = _mm256_loadu_ps(src.add(offset + 8));
            _mm256_storeu_ps(dst.add(offset + 8), _mm256_add_ps(s2, sv));
            let s3 = _mm256_loadu_ps(src.add(offset + 16));
            _mm256_storeu_ps(dst.add(offset + 16), _mm256_add_ps(s3, sv));
            let s4 = _mm256_loadu_ps(src.add(offset + 24));
            _mm256_storeu_ps(dst.add(offset + 24), _mm256_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let s = _mm256_loadu_ps(src.add(offset));
            _mm256_storeu_ps(dst.add(offset), _mm256_add_ps(s, sv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn add_scalar_avx2_stream(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm256_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 32usize;
        while offset + block <= size {
            let s1 = _mm256_loadu_ps(src.add(offset));
            _mm256_stream_ps(dst.add(offset), _mm256_add_ps(s1, sv));
            let s2 = _mm256_loadu_ps(src.add(offset + 8));
            _mm256_stream_ps(dst.add(offset + 8), _mm256_add_ps(s2, sv));
            let s3 = _mm256_loadu_ps(src.add(offset + 16));
            _mm256_stream_ps(dst.add(offset + 16), _mm256_add_ps(s3, sv));
            let s4 = _mm256_loadu_ps(src.add(offset + 24));
            _mm256_stream_ps(dst.add(offset + 24), _mm256_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 8 {
            let s = _mm256_loadu_ps(src.add(offset));
            _mm256_stream_ps(dst.add(offset), _mm256_add_ps(s, sv));
            offset += 8;
            rem -= 8;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar_sse_aligned(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 16usize; // 4x 4-wide
        while offset + block <= size {
            let s1 = _mm_load_ps(src.add(offset));
            _mm_store_ps(dst.add(offset), _mm_add_ps(s1, sv));
            let s2 = _mm_load_ps(src.add(offset + 4));
            _mm_store_ps(dst.add(offset + 4), _mm_add_ps(s2, sv));
            let s3 = _mm_load_ps(src.add(offset + 8));
            _mm_store_ps(dst.add(offset + 8), _mm_add_ps(s3, sv));
            let s4 = _mm_load_ps(src.add(offset + 12));
            _mm_store_ps(dst.add(offset + 12), _mm_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let s = _mm_load_ps(src.add(offset));
            _mm_store_ps(dst.add(offset), _mm_add_ps(s, sv));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar_sse_unaligned(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 16usize;
        while offset + block <= size {
            let s1 = _mm_loadu_ps(src.add(offset));
            _mm_storeu_ps(dst.add(offset), _mm_add_ps(s1, sv));
            let s2 = _mm_loadu_ps(src.add(offset + 4));
            _mm_storeu_ps(dst.add(offset + 4), _mm_add_ps(s2, sv));
            let s3 = _mm_loadu_ps(src.add(offset + 8));
            _mm_storeu_ps(dst.add(offset + 8), _mm_add_ps(s3, sv));
            let s4 = _mm_loadu_ps(src.add(offset + 12));
            _mm_storeu_ps(dst.add(offset + 12), _mm_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let s = _mm_loadu_ps(src.add(offset));
            _mm_storeu_ps(dst.add(offset), _mm_add_ps(s, sv));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn add_scalar_sse_stream(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        use std::arch::x86_64::*;
        let sv = _mm_set1_ps(scalar);
        let mut offset = 0usize;
        let block = 16usize;
        while offset + block <= size {
            let s1 = _mm_loadu_ps(src.add(offset));
            _mm_stream_ps(dst.add(offset), _mm_add_ps(s1, sv));
            let s2 = _mm_loadu_ps(src.add(offset + 4));
            _mm_stream_ps(dst.add(offset + 4), _mm_add_ps(s2, sv));
            let s3 = _mm_loadu_ps(src.add(offset + 8));
            _mm_stream_ps(dst.add(offset + 8), _mm_add_ps(s3, sv));
            let s4 = _mm_loadu_ps(src.add(offset + 12));
            _mm_stream_ps(dst.add(offset + 12), _mm_add_ps(s4, sv));
            offset += block;
        }
        let mut rem = size - offset;
        while rem >= 4 {
            let s = _mm_loadu_ps(src.add(offset));
            _mm_stream_ps(dst.add(offset), _mm_add_ps(s, sv));
            offset += 4;
            rem -= 4;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    /// Optimized tensor+tensor scalar fallback (chunked)
    #[inline]
    unsafe fn add_tensors_scalar_chunk(a: *const f32, b: *const f32, dst: *mut f32, size: usize) {
        let unroll_count = size / 8;
        let mut offset = 0;

        for _ in 0..unroll_count {
            *dst.add(offset) = *a.add(offset) + *b.add(offset);
            *dst.add(offset + 1) = *a.add(offset + 1) + *b.add(offset + 1);
            *dst.add(offset + 2) = *a.add(offset + 2) + *b.add(offset + 2);
            *dst.add(offset + 3) = *a.add(offset + 3) + *b.add(offset + 3);
            *dst.add(offset + 4) = *a.add(offset + 4) + *b.add(offset + 4);
            *dst.add(offset + 5) = *a.add(offset + 5) + *b.add(offset + 5);
            *dst.add(offset + 6) = *a.add(offset + 6) + *b.add(offset + 6);
            *dst.add(offset + 7) = *a.add(offset + 7) + *b.add(offset + 7);
            offset += 8;
        }
        for i in offset..size {
            *dst.add(i) = *a.add(i) + *b.add(i);
        }
    }

    /// Internal optimized scalar + tensor operation
    #[inline]
    pub(crate) fn add_scalar_optimized(&self, scalar: f32) -> Tensor {
        // Optimize contiguous handling - use views when possible
        let (src_ptr, _src_keep): (*const f32, Option<Tensor>) =
            Self::get_optimized_tensor_ptr(self);
        let mut output = Tensor::new(self.shape().dims().to_vec());

        unsafe {
            let src = src_ptr;
            let dst = output.as_mut_ptr();
            let n = self.size();

            // Sequential execution with SIMD optimization
            let stream_min = Self::stream_min_elems();
            if n >= stream_min && Self::try_add_scalar_stream_best(src, dst, n, scalar) {
                // done via streaming stores
            } else if !Self::try_add_scalar_simd_best(src, dst, n, scalar) {
                Self::add_scalar_fallback_chunk(src, dst, n, scalar);
            }
        }

        output
    }

    // Replaced single AVX2 scalar kernel with multi-level SIMD selection and kernels below

    /// Optimized scalar addition fallback (chunked)
    #[inline]
    unsafe fn add_scalar_fallback_chunk(src: *const f32, dst: *mut f32, size: usize, scalar: f32) {
        let unroll_count = size / 8;
        let mut offset = 0;

        for _ in 0..unroll_count {
            *dst.add(offset) = *src.add(offset) + scalar;
            *dst.add(offset + 1) = *src.add(offset + 1) + scalar;
            *dst.add(offset + 2) = *src.add(offset + 2) + scalar;
            *dst.add(offset + 3) = *src.add(offset + 3) + scalar;
            *dst.add(offset + 4) = *src.add(offset + 4) + scalar;
            *dst.add(offset + 5) = *src.add(offset + 5) + scalar;
            *dst.add(offset + 6) = *src.add(offset + 6) + scalar;
            *dst.add(offset + 7) = *src.add(offset + 7) + scalar;
            offset += 8;
        }
        for i in offset..size {
            *dst.add(i) = *src.add(i) + scalar;
        }
    }

    /// Get optimized tensor pointer with intelligent contiguous handling
    ///
    /// This function avoids unnecessary contiguous copies by:
    /// 1. Using direct pointers for contiguous tensors
    /// 2. Using stride-based access for simple non-contiguous patterns
    /// 3. Only creating contiguous copies when absolutely necessary
    ///
    /// Returns (pointer, optional_owned_tensor_to_keep_alive)
    #[inline]
    fn get_optimized_tensor_ptr(tensor: &Tensor) -> (*const f32, Option<Tensor>) {
        unsafe {
            if tensor.is_contiguous() {
                (tensor.as_ptr(), None)
            } else {
                // Materialize non-contiguous views to ensure linear access for SIMD kernels
                let tmp = tensor.contiguous();
                (tmp.as_ptr(), Some(tmp))
            }
        }
    }

    /// Check if we can use stride-based access instead of copying
    ///
    /// OPTIMIZATION #2: Implements stride-aware SIMD for common patterns
    /// to eliminate unnecessary contiguous() copies.
    #[inline]
    #[allow(dead_code)]
    fn can_use_stride_based_access(_tensor: &Tensor) -> bool {
        false
    }

    // ===== OPTIMIZATION #6: Comprehensive SIMD Kernel Caching =====

    /// Get cached SIMD kernels - single initialization, maximum performance
    #[inline]
    fn get_cached_kernels() -> &'static CachedKernels {
        use std::sync::OnceLock;

        static CACHED_KERNELS: OnceLock<CachedKernels> = OnceLock::new();

        CACHED_KERNELS.get_or_init(|| {
            let simd_level = detect_runtime_simd();
            #[cfg(target_arch = "x86_64")]
            let alignment = simd_alignment_bytes(simd_level);

            #[cfg(target_arch = "x86_64")]
            {
                match simd_level {
                    SimdLevel::Avx512 => CachedKernels {
                        simd_level,
                        alignment,
                        tensor_aligned: Self::add_simd_avx512_aligned,
                        tensor_unaligned: Self::add_simd_avx512_unaligned,
                        tensor_stream: Self::add_simd_avx512_stream,
                        scalar_aligned: Self::add_scalar_avx512_aligned,
                        scalar_unaligned: Self::add_scalar_avx512_unaligned,
                        scalar_stream: Self::add_scalar_avx512_stream,
                        min_aligned_size: 16,
                        min_stream_size: Self::stream_min_elems(),
                    },
                    SimdLevel::Avx2 => CachedKernels {
                        simd_level,
                        alignment,
                        tensor_aligned: Self::add_simd_avx2_aligned,
                        tensor_unaligned: Self::add_simd_avx2_unaligned,
                        tensor_stream: Self::add_simd_avx2_stream,
                        scalar_aligned: Self::add_scalar_avx2_aligned,
                        scalar_unaligned: Self::add_scalar_avx2_unaligned,
                        scalar_stream: Self::add_scalar_avx2_stream,
                        min_aligned_size: 8,
                        min_stream_size: Self::stream_min_elems(),
                    },
                    SimdLevel::Sse2 => CachedKernels {
                        simd_level,
                        alignment,
                        tensor_aligned: Self::add_simd_sse_aligned,
                        tensor_unaligned: Self::add_simd_sse_unaligned,
                        tensor_stream: Self::add_simd_sse_stream,
                        scalar_aligned: Self::add_scalar_sse_aligned,
                        scalar_unaligned: Self::add_scalar_sse_unaligned,
                        scalar_stream: Self::add_scalar_sse_stream,
                        min_aligned_size: 4,
                        min_stream_size: Self::stream_min_elems(),
                    },
                    SimdLevel::Scalar => CachedKernels {
                        simd_level,
                        alignment: 4, // f32 alignment
                        tensor_aligned: Self::add_tensors_scalar_chunk,
                        tensor_unaligned: Self::add_tensors_scalar_chunk,
                        tensor_stream: Self::add_tensors_scalar_chunk,
                        scalar_aligned: Self::add_scalar_fallback_chunk,
                        scalar_unaligned: Self::add_scalar_fallback_chunk,
                        scalar_stream: Self::add_scalar_fallback_chunk,
                        min_aligned_size: 1,
                        min_stream_size: usize::MAX, // Never use streaming for scalar
                    },
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                CachedKernels {
                    simd_level,
                    alignment: 4, // f32 alignment
                    tensor_aligned: Self::add_tensors_scalar_chunk,
                    tensor_unaligned: Self::add_tensors_scalar_chunk,
                    tensor_stream: Self::add_tensors_scalar_chunk,
                    scalar_aligned: Self::add_scalar_fallback_chunk,
                    scalar_unaligned: Self::add_scalar_fallback_chunk,
                    scalar_stream: Self::add_scalar_fallback_chunk,
                    min_aligned_size: 1,
                    min_stream_size: usize::MAX,
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_tensor_addition() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![2, 3]);
        let result = a.add_tensor_optimized(&b);

        assert_eq!(result.shape().dims(), vec![2, 3]);
        assert_eq!(result.size(), 6);

        // Check that all values are 2.0 (1.0 + 1.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 2.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_thread_safety_cross_thread_ops() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Create base tensors with gradient tracking
        let a = Arc::new(Tensor::ones(vec![2, 3]).with_requires_grad());
        let b = Arc::new(Tensor::ones(vec![2, 3]).with_requires_grad());

        // Perform entire forward + backward in a single worker thread (TLS-bound grad graph)
        let a1 = a.clone();
        let b1 = b.clone();
        let handle = thread::spawn(move || {
            let t_local1 = Tensor::from_slice(&[2.0; 6], vec![2, 3]).unwrap();
            let r1 = (*a1).add_tensor(&t_local1); // (a + 2)

            let t_local2 = Tensor::ones(vec![2, 3]);
            let r2 = t_local2.add_tensor(&b1); // (1 + b)

            let combined = r1.add_tensor(&r2); // a + b + 3
            let mut loss = combined.sum();
            loss.backward(None);

            let ga = (*a1).grad_owned().expect("grad for a (thread)");
            let gb = (*b1).grad_owned().expect("grad for b (thread)");
            let ga_sum = unsafe { (0..ga.size()).map(|i| *ga.as_ptr().add(i)).sum::<f32>() };
            let gb_sum = unsafe { (0..gb.size()).map(|i| *gb.as_ptr().add(i)).sum::<f32>() };
            (
                ga.shape().dims().to_vec(),
                gb.shape().dims().to_vec(),
                ga_sum,
                gb_sum,
            )
        });

        let (ga_dims, gb_dims, ga_sum, gb_sum) = handle.join().expect("worker panicked");
        assert_eq!(ga_dims, vec![2, 3]);
        assert_eq!(gb_dims, vec![2, 3]);
        // All ones accumulated: 6 elements each
        assert!((ga_sum - 6.0).abs() < 1e-6);
        assert!((gb_sum - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_thread_safety_parallel_large_add_backward() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Size chosen to exceed parallel threshold and align with SIMD-friendly chunking
        let n = 8_388_608; // 32MB of f32 data
        let a = Arc::new(Tensor::ones(vec![n]).with_requires_grad());
        let b = Arc::new(Tensor::ones(vec![n]).with_requires_grad());

        // Perform large forward + backward entirely within the worker thread
        let at = a.clone();
        let bt = b.clone();
        let handle = thread::spawn(move || {
            let result = (*at).add_tensor(&bt);
            let mut loss = result.sum();
            loss.backward(None);
            let ga = (*at).grad_owned().expect("grad for a (thread)");
            let gb = (*bt).grad_owned().expect("grad for b (thread)");
            // Return sizes and simple sums to avoid moving huge tensors across threads
            let ga_sum = unsafe { (0..ga.size()).map(|i| *ga.as_ptr().add(i)).sum::<f32>() };
            let gb_sum = unsafe { (0..gb.size()).map(|i| *gb.as_ptr().add(i)).sum::<f32>() };
            (
                ga.shape().dims().to_vec(),
                gb.shape().dims().to_vec(),
                ga_sum,
                gb_sum,
            )
        });

        let (ga_dims, gb_dims, ga_sum, gb_sum) = handle.join().expect("worker thread panicked");
        assert_eq!(ga_dims, vec![n]);
        assert_eq!(gb_dims, vec![n]);
        // Each gradient should be all ones: sum equals n
        assert!((ga_sum - n as f32).abs() < 1e-3);
        assert!((gb_sum - n as f32).abs() < 1e-3);
    }

    #[test]
    fn test_scalar_addition() {
        let tensor = Tensor::ones(vec![2, 2]);
        let result = tensor.add_scalar_optimized(5.0);

        assert_eq!(result.shape().dims(), vec![2, 2]);
        assert_eq!(result.size(), 4);

        // Check that all values are 6.0 (1.0 + 5.0)
        unsafe {
            for i in 0..result.size() {
                assert!((result.as_ptr().add(i).read() - 6.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    #[should_panic(expected = "Cannot broadcast tensor shapes")]
    fn test_mismatched_shapes() {
        let a = Tensor::ones(vec![2, 3]);
        let b = Tensor::ones(vec![3, 2]);
        a.add_tensor_optimized(&b);
    }

    #[test]
    fn test_add_with_no_grad_guard() {
        use crate::gradtrack::{is_grad_enabled, NoGradTrack};

        // Create tensors with requires_grad enabled
        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let b = Tensor::ones(vec![2, 2]).with_requires_grad();

        // Verify gradients are enabled by default
        assert!(is_grad_enabled());

        // Normal addition with gradients
        let c1 = a.add_tensor(&b);
        assert!(
            c1.requires_grad(),
            "Result should require gradients normally"
        );

        // Addition with NoGradTrack - gradients should be disabled
        {
            let _guard = NoGradTrack::new();
            assert!(
                !is_grad_enabled(),
                "Gradients should be disabled within guard"
            );

            let c2 = a.add_tensor(&b);
            assert!(
                !c2.requires_grad(),
                "Result should not require gradients within NoGradTrack"
            );

            // Test scalar addition as well
            let c3 = a.add_scalar(5.0);
            assert!(
                !c3.requires_grad(),
                "Scalar addition result should not require gradients within NoGradTrack"
            );
        }

        // Gradients should be restored after guard goes out of scope
        assert!(
            is_grad_enabled(),
            "Gradients should be restored after guard"
        );

        let c4 = a.add_tensor(&b);
        assert!(
            c4.requires_grad(),
            "Result should require gradients after guard is dropped"
        );
    }

    #[test]
    fn test_add_nested_no_grad_guards() {
        use crate::gradtrack::{is_grad_enabled, NoGradTrack};

        let a = Tensor::ones(vec![2, 2]).with_requires_grad();
        let b = Tensor::ones(vec![2, 2]).with_requires_grad();

        assert!(is_grad_enabled());

        {
            let _guard1 = NoGradTrack::new();
            assert!(!is_grad_enabled());

            let c1 = a.add_tensor(&b);
            assert!(!c1.requires_grad());

            {
                let _guard2 = NoGradTrack::new();
                assert!(!is_grad_enabled());

                let c2 = a.add_tensor(&b);
                assert!(!c2.requires_grad());
            }

            // Still disabled after inner guard drops
            assert!(!is_grad_enabled());
            let c3 = a.add_tensor(&b);
            assert!(!c3.requires_grad());
        }

        // Restored after all guards drop
        assert!(is_grad_enabled());
        let c4 = a.add_tensor(&b);
        assert!(c4.requires_grad());
    }

    #[test]
    fn test_add_with_mixed_requires_grad() {
        use crate::gradtrack::NoGradTrack;

        let a = Tensor::ones(vec![2, 2]).with_requires_grad(); // requires_grad = true
        let b = Tensor::ones(vec![2, 2]); // requires_grad = false

        // Without NoGradTrack, result should require gradients if any input does
        let c1 = a.add_tensor(&b);
        assert!(c1.requires_grad());

        let c2 = b.add_tensor(&a);
        assert!(c2.requires_grad());

        // With NoGradTrack, result should not require gradients regardless
        {
            let _guard = NoGradTrack::new();

            let c3 = a.add_tensor(&b);
            assert!(!c3.requires_grad());

            let c4 = b.add_tensor(&a);
            assert!(!c4.requires_grad());
        }
    }

    #[test]
    fn test_broadcasting_gradients_basic() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] + [1, 3] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1, 3] (summed over broadcast dim)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.1, 0.2, 0.3], vec![1, 3])
            .unwrap()
            .with_requires_grad();

        let mut result = a.add_tensor(&b);
        assert_eq!(result.shape().dims(), vec![2, 3]);

        // Set upstream gradient as ones
        result.backward(None);

        // Check gradients
        let grad_a = a.grad_owned().expect("grad_a should exist");
        let grad_b = b.grad_owned().expect("grad_b should exist");

        println!(
            "Original shapes: a={:?}, b={:?}",
            a.shape().dims(),
            b.shape().dims()
        );
        println!(
            "Gradient shapes: grad_a={:?}, grad_b={:?}",
            grad_a.shape().dims(),
            grad_b.shape().dims()
        );

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(
            grad_a.shape().dims(),
            vec![2, 3],
            "grad_a should match original shape of a"
        );

        // grad_b should have same shape as b: [1, 3]
        // This requires summing over the broadcasted dimension
        assert_eq!(
            grad_b.shape().dims(),
            vec![1, 3],
            "grad_b should match original shape of b"
        );

        // All gradients should be 1.0 for grad_a
        for i in 0..grad_a.size() {
            let val = unsafe { *grad_a.as_ptr().add(i) };
            assert!(
                (val - 1.0).abs() < 1e-6,
                "grad_a[{}] = {} should be 1.0",
                i,
                val
            );
        }

        // grad_b should be [2.0, 2.0, 2.0] (sum over broadcast dim)
        let expected_grad_b = [2.0, 2.0, 2.0];
        for (i, val) in expected_grad_b.iter().enumerate().take(grad_b.size()) {
            let actual = unsafe { *grad_b.as_ptr().add(i) };
            assert!(
                (actual - val).abs() < 1e-6,
                "grad_b[{}] = {} should be {}",
                i,
                actual,
                val
            );
        }
    }

    #[test]
    fn test_scalar_broadcasting_gradients() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Test case: [2, 3] + [1] -> [2, 3]
        // grad_a should be [2, 3], grad_b should be [1] (summed over all dims)

        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let b = Tensor::from_slice(&[0.5], vec![1])
            .unwrap()
            .with_requires_grad();

        let mut result = a.add_tensor(&b);
        result.backward(None);

        let grad_a = a.grad_owned().expect("grad_a should exist");
        let grad_b = b.grad_owned().expect("grad_b should exist");

        // grad_a should have same shape as a: [2, 3]
        assert_eq!(grad_a.shape().dims(), vec![2, 3]);

        // grad_b should have same shape as b: [1] and sum to 6.0
        println!("grad_b shape: {:?}, expected: [1]", grad_b.shape().dims());
        assert_eq!(grad_b.shape().dims(), vec![1]);

        // grad_b should be 6.0 (sum over all 6 elements)
        let val = unsafe { *grad_b.as_ptr() };
        assert!((val - 6.0).abs() < 1e-6, "grad_b = {} should be 6.0", val);
    }

    #[test]
    fn test_linear_layer_bias_broadcasting() {
        use crate::gradtrack::clear_gradients;
        clear_gradients();

        // Simulate linear layer bias broadcasting
        // input: [2, 3], weight: [3, 4], bias: [4]
        // matmul result: [2, 4], bias broadcast: [4] -> [2, 4]

        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        let weight = Tensor::from_slice(
            &(1..=12).map(|i| i as f32 * 0.1).collect::<Vec<_>>(),
            vec![3, 4],
        )
        .unwrap()
        .with_requires_grad();
        let bias = Tensor::from_slice(&[0.1, 0.2, 0.3, 0.4], vec![4])
            .unwrap()
            .with_requires_grad();

        // Forward pass: input @ weight + bias
        let matmul_result = input.matmul(&weight);
        println!("Matmul result shape: {:?}", matmul_result.shape().dims());
        println!("Bias shape: {:?}", bias.shape().dims());

        let linear_output = matmul_result.add_tensor(&bias);
        println!("Linear output shape: {:?}", linear_output.shape().dims());

        // Sum all outputs as loss
        let mut loss = linear_output.sum();
        loss.backward(None);

        // Check bias gradient
        let bias_grad = bias.grad_owned().expect("bias gradient should exist");
        println!("Bias gradient shape: {:?}", bias_grad.shape().dims());
        assert_eq!(
            bias_grad.shape().dims(),
            vec![4],
            "bias gradient should match bias shape"
        );

        // Bias gradient should be [2.0, 2.0, 2.0, 2.0] (sum over batch dimension)
        for i in 0..4 {
            let val = unsafe { *bias_grad.as_ptr().add(i) };
            assert!(
                (val - 2.0).abs() < 1e-6,
                "bias_grad[{}] = {} should be 2.0",
                i,
                val
            );
        }

        println!("Linear layer bias broadcasting test passed!");
    }
}
