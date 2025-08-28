//! SIMD-optimized matrix multiplication kernels
//!
//! This module contains high-performance computational kernels for matrix multiplication,
//! with AVX2 SIMD optimizations and scalar fallbacks. The implementation provides
//! intelligent dispatch to size-specific kernels optimized for common machine learning
//! matrix dimensions.
//!
//! # Key Features
//!
//! - **AVX2 SIMD Optimization**: 8x vectorization for compatible hardware
//! - **Intelligent Dispatch**: Size-specific kernels for optimal performance
//! - **Cache Optimization**: Block sizes tuned for L1/L2/L3 cache hierarchies
//! - **Memory Bandwidth**: Aggressive prefetching for large matrix operations
//! - **Scalar Fallbacks**: Optimized scalar implementations for non-SIMD hardware
//! - **ML Workload Tuning**: Optimized for common neural network matrix sizes
//! - **Memory Safety**: Safe handling of uninitialized memory with accumulation
//!
//! # Architecture
//!
//! The kernel system uses ML-optimized dispatch to select the most efficient implementation
//! based on matrix dimensions commonly found in neural network workloads:
//! - **Small matrices (16-64 elements)**: Direct computation with minimal overhead
//! - **Medium-small matrices (64-128 elements)**: 2x2 micro-kernel blocking
//! - **Medium matrices (128-256 elements)**: 4x8 micro-kernel with prefetching
//! - **Large matrices (256-512 elements)**: Cache blocking with aggressive prefetching
//! - **Extra-large matrices (512+ elements)**: Hierarchical blocking for memory bandwidth
//!
//! # Performance Characteristics
//!
//! - **AVX2 acceleration**: 8x SIMD operations for compatible hardware
//! - **Cache optimization**: Block sizes tuned for L1/L2/L3 cache hierarchies
//! - **Memory bandwidth**: Aggressive prefetching for large matrix operations
//! - **Scalar fallbacks**: Optimized scalar implementations for non-SIMD hardware
//! - **ML workload tuning**: Optimized for common neural network matrix sizes
//! - **Memory safety**: Seamless integration with uninitialized tensor allocation
//!
//! # Memory Safety
//!
//! All kernels handle memory initialization safely through the `is_first_block` parameter,
//! which determines whether to initialize or accumulate results. This approach works
//! seamlessly with `Tensor::new_uninitialized` for performance-critical allocations.
//!
//! # Implementation Details
//!
//! The kernel system provides multiple specialized implementations:
//!
//! - **Small Matrix Kernels**: Direct computation with minimal overhead for matrices <64 elements
//! - **Medium Matrix Kernels**: Cache-aware blocking with prefetching for matrices 64-256 elements
//! - **Large Matrix Kernels**: Memory bandwidth optimization with hierarchical blocking for matrices 256+ elements
//! - **Scalar Fallbacks**: Optimized scalar implementations for non-SIMD hardware
//! - **Blocked Algorithm**: Three-level blocked approach with ML-specific optimizations

use super::config::MatmulConfig;

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Cache-aligned buffer for matrix multiplication operations
///
/// Provides pre-allocated buffers for matrix multiplication kernels to avoid
/// repeated memory allocations during computation. The buffer is cache-line
/// aligned for optimal memory access performance.
///
/// # Fields
///
/// * `packed_b` - Pre-allocated buffer for packed matrix data
///
/// # Performance Characteristics
///
/// - **Cache-line aligned**: 64-byte alignment for optimal memory access
/// - **Pre-allocated**: Avoids runtime allocation overhead
/// - **Reusable**: Buffer can be resized for different matrix dimensions
#[repr(align(64))] // Cache line aligned for modern CPUs
pub struct MatmulBuffers {
    pub packed_b: Vec<f32>,
}

impl MatmulBuffers {
    pub fn new() -> Self {
        Self {
            packed_b: vec![0.0f32; 1024], // Pre-allocate reasonable buffer size
        }
    }
}

impl Default for MatmulBuffers {
    fn default() -> Self {
        Self::new()
    }
}

impl MatmulBuffers {
    #[allow(unused)]
    pub fn ensure_capacity(&mut self, required_size: usize) {
        if self.packed_b.len() < required_size {
            self.packed_b.resize(required_size, 0.0f32);
        }
    }
}

/// ML-optimized matrix multiplication kernel with intelligent dispatch
///
/// Selects the most efficient matrix multiplication implementation based on matrix
/// dimensions and hardware capabilities. The dispatch system targets common machine
/// learning workload sizes for optimal performance in neural network operations.
///
/// # Arguments
///
/// * `a_ptr` - Pointer to first matrix data (row-major layout)
/// * `b_ptr` - Pointer to second matrix data (row-major layout)
/// * `c_ptr` - Pointer to result matrix data (row-major layout)
/// * `m` - Number of rows in first matrix and result
/// * `n` - Number of columns in second matrix and result
/// * `k` - Inner dimension (columns of first matrix, rows of second matrix)
/// * `is_first_block` - Whether this is the first k-block (initializes vs accumulates)
///
/// # Safety
///
/// Requires valid pointers with sufficient memory for the given dimensions.
/// Memory layout must be contiguous in row-major order. Memory initialization
/// and accumulation are handled safely within each specialized kernel.
///
/// # Performance Characteristics
///
/// - **Intelligent dispatch**: Selects optimal kernel based on matrix dimensions
/// - **Hardware detection**: Automatically uses AVX2 when available
/// - **ML workload optimization**: Tuned for common neural network matrix sizes
#[inline]
pub unsafe fn matrix_multiply_kernel(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Intelligent dispatch based on matrix characteristics for ML workloads
            dispatch_ml_optimized_kernel_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        } else {
            // Fallback with size-aware scalar optimizations
            dispatch_ml_optimized_kernel_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        dispatch_ml_optimized_kernel_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
    }
}

/// ML-optimized AVX2 dispatch system for targeted matrix sizes
///
/// Selects the most efficient AVX2 kernel based on matrix dimensions commonly
/// found in ML workloads. Each kernel is tuned for specific size ranges to
/// maximize cache efficiency and SIMD utilization.
///
/// # Safety
///
/// Requires AVX2 support. All kernels handle memory initialization and accumulation safely.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn dispatch_ml_optimized_kernel_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let max_dim = m.max(n).max(k);
    let min_dim = m.min(n).min(k);

    // Dispatch based on ML-common matrix size categories
    match max_dim {
        0..=64 if min_dim <= 32 => {
            // Small matrices: optimized for L1 cache and minimal overhead
            matrix_multiply_small_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=128 if min_dim <= 64 => {
            // Medium-small matrices: balanced for L1/L2 cache
            matrix_multiply_medium_small_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=256 if min_dim <= 128 => {
            // Medium matrices: optimized for L2 cache with aggressive prefetching
            matrix_multiply_medium_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=512 if min_dim <= 256 => {
            // Large matrices: memory bandwidth optimized with cache blocking
            matrix_multiply_large_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        _ => {
            // Very large matrices: aggressive blocking and prefetching
            matrix_multiply_xlarge_avx2(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
    }
}

/// ML-optimized scalar dispatch system for targeted matrix sizes
///
/// Provides efficient scalar fallbacks with size-specific optimizations
/// for when AVX2 is not available.
///
/// # Safety
///
/// All kernels handle memory initialization and accumulation safely.
#[inline]
unsafe fn dispatch_ml_optimized_kernel_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let max_dim = m.max(n).max(k);
    let min_dim = m.min(n).min(k);

    match max_dim {
        0..=64 if min_dim <= 32 => {
            matrix_multiply_small_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=128 if min_dim <= 64 => {
            matrix_multiply_medium_small_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=256 if min_dim <= 128 => {
            matrix_multiply_medium_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        0..=512 if min_dim <= 256 => {
            matrix_multiply_large_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
        _ => {
            matrix_multiply_xlarge_scalar(a_ptr, b_ptr, c_ptr, m, n, k, is_first_block);
        }
    }
}

// ===== Small Matrix Kernels (16-64 elements) =====

/// Small matrix AVX2 kernel optimized for L1 cache efficiency
///
/// Optimized for matrices with dimensions typically 16-64. Uses minimal blocking
/// and focuses on register reuse to maximize L1 cache hit rate.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_small_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // For small matrices, use direct computation with minimal overhead
    let simd_width = 8;
    let n_simd = n / simd_width * simd_width;

    for i in 0..m {
        // Process 8 columns at a time
        let mut j = 0;
        while j < n_simd {
            let mut sum_vec = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add(i * n + j))
            };

            // Unrolled k-loop for better register utilization (common case k=16,32,64)
            let mut p = 0;
            let k_unroll = k / 4 * 4;
            while p < k_unroll {
                // Unroll 4 iterations for better instruction level parallelism
                let a_val0 = _mm256_set1_ps(*a_ptr.add(i * k + p));
                let a_val1 = _mm256_set1_ps(*a_ptr.add(i * k + p + 1));
                let a_val2 = _mm256_set1_ps(*a_ptr.add(i * k + p + 2));
                let a_val3 = _mm256_set1_ps(*a_ptr.add(i * k + p + 3));

                let b_vec0 = _mm256_loadu_ps(b_ptr.add(p * n + j));
                let b_vec1 = _mm256_loadu_ps(b_ptr.add((p + 1) * n + j));
                let b_vec2 = _mm256_loadu_ps(b_ptr.add((p + 2) * n + j));
                let b_vec3 = _mm256_loadu_ps(b_ptr.add((p + 3) * n + j));

                sum_vec = _mm256_fmadd_ps(a_val0, b_vec0, sum_vec);
                sum_vec = _mm256_fmadd_ps(a_val1, b_vec1, sum_vec);
                sum_vec = _mm256_fmadd_ps(a_val2, b_vec2, sum_vec);
                sum_vec = _mm256_fmadd_ps(a_val3, b_vec3, sum_vec);
                p += 4;
            }

            // Handle remaining k elements
            while p < k {
                let a_val = _mm256_set1_ps(*a_ptr.add(i * k + p));
                let b_vec = _mm256_loadu_ps(b_ptr.add(p * n + j));
                sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                p += 1;
            }

            _mm256_storeu_ps(c_ptr.add(i * n + j), sum_vec);
            j += simd_width;
        }

        // Handle remaining columns with optimized scalar code
        for j in n_simd..n {
            let mut sum = if is_first_block {
                0.0
            } else {
                *c_ptr.add(i * n + j)
            };

            // Unrolled scalar loop
            let mut p = 0;
            let k_unroll = k / 4 * 4;
            while p < k_unroll {
                sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                sum += *a_ptr.add(i * k + p + 1) * *b_ptr.add((p + 1) * n + j);
                sum += *a_ptr.add(i * k + p + 2) * *b_ptr.add((p + 2) * n + j);
                sum += *a_ptr.add(i * k + p + 3) * *b_ptr.add((p + 3) * n + j);
                p += 4;
            }

            while p < k {
                sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                p += 1;
            }

            *c_ptr.add(i * n + j) = sum;
        }
    }
}

/// Medium-small matrix AVX2 kernel (64-128 elements)
///
/// Optimized for matrices commonly found in smaller neural network layers.
/// Uses 2x2 micro-kernel blocking for better instruction throughput.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_medium_small_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let simd_width = 8;
    let n_simd = n / simd_width * simd_width;

    // Process 2 rows at a time for better cache utilization
    let mut i = 0;
    let m_pairs = m / 2 * 2;

    while i < m_pairs {
        let mut j = 0;
        while j < n_simd {
            let mut sum_vec0 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add(i * n + j))
            };
            let mut sum_vec1 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add((i + 1) * n + j))
            };

            // Inner product with 2x8 micro-kernel
            for p in 0..k {
                let a_val0 = _mm256_set1_ps(*a_ptr.add(i * k + p));
                let a_val1 = _mm256_set1_ps(*a_ptr.add((i + 1) * k + p));
                let b_vec = _mm256_loadu_ps(b_ptr.add(p * n + j));

                sum_vec0 = _mm256_fmadd_ps(a_val0, b_vec, sum_vec0);
                sum_vec1 = _mm256_fmadd_ps(a_val1, b_vec, sum_vec1);
            }

            _mm256_storeu_ps(c_ptr.add(i * n + j), sum_vec0);
            _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), sum_vec1);
            j += simd_width;
        }

        // Handle remaining columns
        for j in n_simd..n {
            for row_offset in 0..2 {
                let row = i + row_offset;
                let mut sum = if is_first_block {
                    0.0
                } else {
                    *c_ptr.add(row * n + j)
                };

                for p in 0..k {
                    sum += *a_ptr.add(row * k + p) * *b_ptr.add(p * n + j);
                }

                *c_ptr.add(row * n + j) = sum;
            }
        }
        i += 2;
    }

    // Handle remaining rows
    for i in m_pairs..m {
        matrix_multiply_small_avx2(
            a_ptr.add(i * k),
            b_ptr,
            c_ptr.add(i * n),
            1,
            n,
            k,
            is_first_block,
        );
    }
}

/// Medium matrix AVX2 kernel (128-256 elements)
///
/// Optimized for common hidden layer sizes. Uses cache-aware blocking
/// with prefetching for memory bandwidth optimization.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_medium_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let simd_width = 8;
    let n_simd = n / simd_width * simd_width;

    // Use 4x8 micro-kernels for better instruction throughput
    let mut i = 0;
    let m_quads = m / 4 * 4;

    while i < m_quads {
        let mut j = 0;
        while j < n_simd {
            let mut sum_vec0 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add(i * n + j))
            };
            let mut sum_vec1 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add((i + 1) * n + j))
            };
            let mut sum_vec2 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add((i + 2) * n + j))
            };
            let mut sum_vec3 = if is_first_block {
                _mm256_setzero_ps()
            } else {
                _mm256_loadu_ps(c_ptr.add((i + 3) * n + j))
            };

            // 4x8 micro-kernel with prefetching
            for p in 0..k {
                // Prefetch next iteration's data
                if p + 8 < k {
                    #[cfg(target_arch = "x86_64")]
                    {
                        let prefetch_addr = b_ptr.add((p + 8) * n + j) as *const i8;
                        std::arch::x86_64::_mm_prefetch(
                            prefetch_addr,
                            std::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }

                let a_val0 = _mm256_set1_ps(*a_ptr.add(i * k + p));
                let a_val1 = _mm256_set1_ps(*a_ptr.add((i + 1) * k + p));
                let a_val2 = _mm256_set1_ps(*a_ptr.add((i + 2) * k + p));
                let a_val3 = _mm256_set1_ps(*a_ptr.add((i + 3) * k + p));
                let b_vec = _mm256_loadu_ps(b_ptr.add(p * n + j));

                sum_vec0 = _mm256_fmadd_ps(a_val0, b_vec, sum_vec0);
                sum_vec1 = _mm256_fmadd_ps(a_val1, b_vec, sum_vec1);
                sum_vec2 = _mm256_fmadd_ps(a_val2, b_vec, sum_vec2);
                sum_vec3 = _mm256_fmadd_ps(a_val3, b_vec, sum_vec3);
            }

            _mm256_storeu_ps(c_ptr.add(i * n + j), sum_vec0);
            _mm256_storeu_ps(c_ptr.add((i + 1) * n + j), sum_vec1);
            _mm256_storeu_ps(c_ptr.add((i + 2) * n + j), sum_vec2);
            _mm256_storeu_ps(c_ptr.add((i + 3) * n + j), sum_vec3);
            j += simd_width;
        }

        // Handle remaining columns
        for j in n_simd..n {
            for row_offset in 0..4 {
                let row = i + row_offset;
                let mut sum = if is_first_block {
                    0.0
                } else {
                    *c_ptr.add(row * n + j)
                };

                for p in 0..k {
                    sum += *a_ptr.add(row * k + p) * *b_ptr.add(p * n + j);
                }

                *c_ptr.add(row * n + j) = sum;
            }
        }
        i += 4;
    }

    // Handle remaining rows
    for i in m_quads..m {
        matrix_multiply_small_avx2(
            a_ptr.add(i * k),
            b_ptr,
            c_ptr.add(i * n),
            1,
            n,
            k,
            is_first_block,
        );
    }
}

/// Large matrix AVX2 kernel (256-512 elements)  
///
/// Optimized for larger neural network layers with cache blocking
/// and aggressive prefetching strategies.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_large_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // Use blocked approach for large matrices to optimize cache usage
    let block_size = 64; // Optimize for L2 cache
    let simd_width = 8;

    for i_block in (0..m).step_by(block_size) {
        let i_end = (i_block + block_size).min(m);
        for j_block in (0..n).step_by(block_size) {
            let j_end = (j_block + block_size).min(n);

            // Process block with optimized inner kernel
            for i in i_block..i_end {
                let mut j = j_block;
                let j_simd_end = ((j_end - j_block) / simd_width * simd_width) + j_block;

                while j < j_simd_end {
                    let mut sum_vec = if is_first_block {
                        _mm256_setzero_ps()
                    } else {
                        _mm256_loadu_ps(c_ptr.add(i * n + j))
                    };

                    // Aggressive prefetching for memory bandwidth
                    for p in 0..k {
                        if p % 4 == 0 && p + 16 < k {
                            let prefetch_addr_a = a_ptr.add(i * k + p + 16) as *const i8;
                            let prefetch_addr_b = b_ptr.add((p + 16) * n + j) as *const i8;
                            std::arch::x86_64::_mm_prefetch(
                                prefetch_addr_a,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                            std::arch::x86_64::_mm_prefetch(
                                prefetch_addr_b,
                                std::arch::x86_64::_MM_HINT_T0,
                            );
                        }

                        let a_val = _mm256_set1_ps(*a_ptr.add(i * k + p));
                        let b_vec = _mm256_loadu_ps(b_ptr.add(p * n + j));
                        sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                    }

                    _mm256_storeu_ps(c_ptr.add(i * n + j), sum_vec);
                    j += simd_width;
                }

                // Handle remaining columns in block
                for j in j_simd_end..j_end {
                    let mut sum = if is_first_block {
                        0.0
                    } else {
                        *c_ptr.add(i * n + j)
                    };

                    for p in 0..k {
                        sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                    }

                    *c_ptr.add(i * n + j) = sum;
                }
            }
        }
    }
}

/// Extra-large matrix AVX2 kernel (512+ elements)
///
/// Optimized for very large matrices with hierarchical blocking
/// and memory bandwidth optimization.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn matrix_multiply_xlarge_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // Hierarchical blocking for very large matrices
    let l3_block = 256; // L3 cache blocking
    let l2_block = 64; // L2 cache blocking
    let simd_width = 8;

    for i_l3 in (0..m).step_by(l3_block) {
        let i_l3_end = (i_l3 + l3_block).min(m);
        for j_l3 in (0..n).step_by(l3_block) {
            let j_l3_end = (j_l3 + l3_block).min(n);
            for k_l3 in (0..k).step_by(l3_block) {
                let k_l3_end = (k_l3 + l3_block).min(k);

                // L2 blocking within L3 blocks
                for i_l2 in (i_l3..i_l3_end).step_by(l2_block) {
                    let i_l2_end = (i_l2 + l2_block).min(i_l3_end);
                    for j_l2 in (j_l3..j_l3_end).step_by(l2_block) {
                        let j_l2_end = (j_l2 + l2_block).min(j_l3_end);

                        // Inner kernel with aggressive prefetching
                        for i in i_l2..i_l2_end {
                            let mut j = j_l2;
                            let j_simd_end = ((j_l2_end - j_l2) / simd_width * simd_width) + j_l2;

                            while j < j_simd_end {
                                let mut sum_vec = if is_first_block && k_l3 == 0 {
                                    _mm256_setzero_ps()
                                } else {
                                    _mm256_loadu_ps(c_ptr.add(i * n + j))
                                };

                                for p in k_l3..k_l3_end {
                                    // Multi-level prefetching
                                    if p % 8 == 0 && p + 32 < k_l3_end {
                                        let prefetch_addr =
                                            b_ptr.add((p + 32) * n + j) as *const i8;
                                        std::arch::x86_64::_mm_prefetch(
                                            prefetch_addr,
                                            std::arch::x86_64::_MM_HINT_T1,
                                        );
                                    }

                                    let a_val = _mm256_set1_ps(*a_ptr.add(i * k + p));
                                    let b_vec = _mm256_loadu_ps(b_ptr.add(p * n + j));
                                    sum_vec = _mm256_fmadd_ps(a_val, b_vec, sum_vec);
                                }

                                _mm256_storeu_ps(c_ptr.add(i * n + j), sum_vec);
                                j += simd_width;
                            }

                            // Handle remaining columns
                            for j in j_simd_end..j_l2_end {
                                let mut sum = if is_first_block && k_l3 == 0 {
                                    0.0
                                } else {
                                    *c_ptr.add(i * n + j)
                                };

                                for p in k_l3..k_l3_end {
                                    sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                                }

                                *c_ptr.add(i * n + j) = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

// ===== Scalar Fallback Kernels =====

/// Small matrix scalar kernel optimized for minimal overhead
#[inline]
unsafe fn matrix_multiply_small_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // Direct computation with 4x unrolling for small matrices
    for i in 0..m {
        for j in 0..n {
            let mut sum = if is_first_block {
                0.0
            } else {
                *c_ptr.add(i * n + j)
            };

            // Unroll k-loop by 4 for better instruction-level parallelism
            let mut p = 0;
            let k_unroll = k / 4 * 4;
            while p < k_unroll {
                sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                sum += *a_ptr.add(i * k + p + 1) * *b_ptr.add((p + 1) * n + j);
                sum += *a_ptr.add(i * k + p + 2) * *b_ptr.add((p + 2) * n + j);
                sum += *a_ptr.add(i * k + p + 3) * *b_ptr.add((p + 3) * n + j);
                p += 4;
            }

            // Handle remaining elements
            while p < k {
                sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                p += 1;
            }

            *c_ptr.add(i * n + j) = sum;
        }
    }
}

/// Medium-small matrix scalar kernel with 2x2 blocking
#[inline]
unsafe fn matrix_multiply_medium_small_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // Use 2x2 micro-kernel blocking for better cache utilization
    let mut i = 0;
    let m_pairs = m / 2 * 2;

    while i < m_pairs {
        let mut j = 0;
        let n_pairs = n / 2 * 2;

        // Process 2x2 blocks
        while j < n_pairs {
            let mut sum00 = if is_first_block {
                0.0
            } else {
                *c_ptr.add(i * n + j)
            };
            let mut sum01 = if is_first_block {
                0.0
            } else {
                *c_ptr.add(i * n + j + 1)
            };
            let mut sum10 = if is_first_block {
                0.0
            } else {
                *c_ptr.add((i + 1) * n + j)
            };
            let mut sum11 = if is_first_block {
                0.0
            } else {
                *c_ptr.add((i + 1) * n + j + 1)
            };

            for p in 0..k {
                let a0 = *a_ptr.add(i * k + p);
                let a1 = *a_ptr.add((i + 1) * k + p);
                let b0 = *b_ptr.add(p * n + j);
                let b1 = *b_ptr.add(p * n + j + 1);

                sum00 += a0 * b0;
                sum01 += a0 * b1;
                sum10 += a1 * b0;
                sum11 += a1 * b1;
            }

            *c_ptr.add(i * n + j) = sum00;
            *c_ptr.add(i * n + j + 1) = sum01;
            *c_ptr.add((i + 1) * n + j) = sum10;
            *c_ptr.add((i + 1) * n + j + 1) = sum11;

            j += 2;
        }

        // Handle remaining columns
        for j in n_pairs..n {
            for row_offset in 0..2 {
                let row = i + row_offset;
                let mut sum = if is_first_block {
                    0.0
                } else {
                    *c_ptr.add(row * n + j)
                };

                for p in 0..k {
                    sum += *a_ptr.add(row * k + p) * *b_ptr.add(p * n + j);
                }

                *c_ptr.add(row * n + j) = sum;
            }
        }
        i += 2;
    }

    // Handle remaining rows
    for i in m_pairs..m {
        matrix_multiply_small_scalar(
            a_ptr.add(i * k),
            b_ptr,
            c_ptr.add(i * n),
            1,
            n,
            k,
            is_first_block,
        );
    }
}

/// Medium matrix scalar kernel with 4x4 blocking
#[inline]
unsafe fn matrix_multiply_medium_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    // Use 4x4 micro-kernel blocking
    let mut i = 0;
    let m_quads = m / 4 * 4;

    while i < m_quads {
        let mut j = 0;
        let n_quads = n / 4 * 4;

        while j < n_quads {
            // 4x4 register block
            let mut sums = [[0.0f32; 4]; 4];

            // Initialize with existing values if not first block
            if !is_first_block {
                for (row_idx, row) in sums.iter_mut().enumerate() {
                    for (col_idx, sum_val) in row.iter_mut().enumerate() {
                        *sum_val = *c_ptr.add((i + row_idx) * n + j + col_idx);
                    }
                }
            }

            // Compute 4x4 block
            for p in 0..k {
                let a_vals = [
                    *a_ptr.add(i * k + p),
                    *a_ptr.add((i + 1) * k + p),
                    *a_ptr.add((i + 2) * k + p),
                    *a_ptr.add((i + 3) * k + p),
                ];
                let b_vals = [
                    *b_ptr.add(p * n + j),
                    *b_ptr.add(p * n + j + 1),
                    *b_ptr.add(p * n + j + 2),
                    *b_ptr.add(p * n + j + 3),
                ];

                for (row_idx, &a_val) in a_vals.iter().enumerate() {
                    for (col_idx, &b_val) in b_vals.iter().enumerate() {
                        sums[row_idx][col_idx] += a_val * b_val;
                    }
                }
            }

            // Store results
            for (row_idx, row) in sums.iter().enumerate() {
                for (col_idx, &sum_val) in row.iter().enumerate() {
                    *c_ptr.add((i + row_idx) * n + j + col_idx) = sum_val;
                }
            }

            j += 4;
        }

        // Handle remaining columns
        for j in n_quads..n {
            for row_offset in 0..4 {
                let row = i + row_offset;
                let mut sum = if is_first_block {
                    0.0
                } else {
                    *c_ptr.add(row * n + j)
                };

                for p in 0..k {
                    sum += *a_ptr.add(row * k + p) * *b_ptr.add(p * n + j);
                }

                *c_ptr.add(row * n + j) = sum;
            }
        }
        i += 4;
    }

    // Handle remaining rows
    for i in m_quads..m {
        matrix_multiply_small_scalar(
            a_ptr.add(i * k),
            b_ptr,
            c_ptr.add(i * n),
            1,
            n,
            k,
            is_first_block,
        );
    }
}

/// Large matrix scalar kernel with cache blocking
#[inline]
unsafe fn matrix_multiply_large_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let block_size = 64; // L2 cache optimized

    for i_block in (0..m).step_by(block_size) {
        let i_end = (i_block + block_size).min(m);
        for j_block in (0..n).step_by(block_size) {
            let j_end = (j_block + block_size).min(n);

            // Process block using medium kernel
            for i in i_block..i_end {
                for j in j_block..j_end {
                    let mut sum = if is_first_block {
                        0.0
                    } else {
                        *c_ptr.add(i * n + j)
                    };

                    // Unrolled inner loop
                    let mut p = 0;
                    let k_unroll = k / 8 * 8;
                    while p < k_unroll {
                        sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                        sum += *a_ptr.add(i * k + p + 1) * *b_ptr.add((p + 1) * n + j);
                        sum += *a_ptr.add(i * k + p + 2) * *b_ptr.add((p + 2) * n + j);
                        sum += *a_ptr.add(i * k + p + 3) * *b_ptr.add((p + 3) * n + j);
                        sum += *a_ptr.add(i * k + p + 4) * *b_ptr.add((p + 4) * n + j);
                        sum += *a_ptr.add(i * k + p + 5) * *b_ptr.add((p + 5) * n + j);
                        sum += *a_ptr.add(i * k + p + 6) * *b_ptr.add((p + 6) * n + j);
                        sum += *a_ptr.add(i * k + p + 7) * *b_ptr.add((p + 7) * n + j);
                        p += 8;
                    }

                    while p < k {
                        sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                        p += 1;
                    }

                    *c_ptr.add(i * n + j) = sum;
                }
            }
        }
    }
}

/// Extra-large matrix scalar kernel with hierarchical blocking
#[inline]
unsafe fn matrix_multiply_xlarge_scalar(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    is_first_block: bool,
) {
    let l3_block = 256; // L3 cache blocking
    let l2_block = 64; // L2 cache blocking

    for i_l3 in (0..m).step_by(l3_block) {
        let i_l3_end = (i_l3 + l3_block).min(m);
        for j_l3 in (0..n).step_by(l3_block) {
            let j_l3_end = (j_l3 + l3_block).min(n);
            for k_l3 in (0..k).step_by(l3_block) {
                let k_l3_end = (k_l3 + l3_block).min(k);

                // L2 blocking within L3 blocks
                for i_l2 in (i_l3..i_l3_end).step_by(l2_block) {
                    let i_l2_end = (i_l2 + l2_block).min(i_l3_end);
                    for j_l2 in (j_l3..j_l3_end).step_by(l2_block) {
                        let j_l2_end = (j_l2 + l2_block).min(j_l3_end);

                        // Inner computation
                        for i in i_l2..i_l2_end {
                            for j in j_l2..j_l2_end {
                                let mut sum = if is_first_block && k_l3 == 0 {
                                    0.0
                                } else {
                                    *c_ptr.add(i * n + j)
                                };

                                for p in k_l3..k_l3_end {
                                    sum += *a_ptr.add(i * k + p) * *b_ptr.add(p * n + j);
                                }

                                *c_ptr.add(i * n + j) = sum;
                            }
                        }
                    }
                }
            }
        }
    }
}

/// ML-optimized blocked matrix multiplication with enhanced memory safety
///
/// Performs matrix multiplication using blocked algorithms optimized for machine learning
/// workloads. This implementation integrates with `Tensor::new_uninitialized` safety
/// mechanisms to ensure proper memory initialization and accumulation.
///
/// # Arguments
///
/// * `a_ptr` - Pointer to first matrix data (row-major layout)
/// * `b_ptr` - Pointer to second matrix data (row-major layout)
/// * `c_ptr` - Pointer to result matrix data (row-major layout)
/// * `m` - Number of rows in first matrix and result
/// * `n` - Number of columns in second matrix and result
/// * `k` - Inner dimension (columns of first matrix, rows of second matrix)
/// * `config` - Configuration with optimal block sizes and kernel type
///
/// # Algorithm
///
/// Uses a three-level blocked approach with ML-specific optimizations:
/// 1. **ML-aware dispatch**: Size-specific kernels for common matrix dimensions
/// 2. **Cache-optimized blocking**: Block sizes tuned for L1/L2/L3 cache hierarchies
/// 3. **Safe accumulation**: Proper initialization and accumulation across k-blocks
///
/// # Memory Safety
///
/// The function assumes that the result matrix `c_ptr` was allocated using
/// `Tensor::new_uninitialized` and handles initialization safely:
/// - **First k-block**: Initializes values (is_first_block = true)
/// - **Subsequent k-blocks**: Accumulates values (is_first_block = false)
/// - This avoids reading uninitialized memory while maintaining performance
///
/// # Safety
///
/// Assumes all pointers are valid and within bounds. The caller must ensure
/// that `c_ptr` points to properly allocated memory (typically via `Tensor::new_uninitialized`).
///
/// # Performance Characteristics
///
/// - **Blocked computation**: Optimized for cache efficiency and memory bandwidth
/// - **Kernel dispatch**: Intelligent selection based on matrix dimensions
/// - **Memory safety**: Seamless integration with uninitialized tensor allocation
#[allow(clippy::too_many_arguments)]
pub unsafe fn matrix_multiply_blocked(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    config: &MatmulConfig,
) {
    // Use reasonable block sizes for cache efficiency
    let block_m = config.block_m.min(m);
    let block_n = config.block_n.min(n);
    let block_k = config.block_k.min(k);

    // Enhanced safety: Work with Tensor::new_uninitialized memory safely
    // The first k-block will initialize values, subsequent blocks will accumulate
    // This approach avoids touching potentially uninitialized memory while maintaining performance

    // Blocked algorithm with safe initialization/accumulation pattern
    for kk in (0..k).step_by(block_k) {
        let kb = (kk + block_k).min(k) - kk;
        let is_first_k_block = kk == 0; // First block initializes, subsequent blocks accumulate

        for ii in (0..m).step_by(block_m) {
            let ib = (ii + block_m).min(m) - ii;

            for jj in (0..n).step_by(block_n) {
                let jb = (jj + block_n).min(n) - jj;

                // Use the ML-optimized kernel dispatch for this block
                matrix_multiply_kernel(
                    a_ptr.add(ii * k + kk),
                    b_ptr.add(kk * n + jj),
                    c_ptr.add(ii * n + jj),
                    ib,
                    jb,
                    kb,
                    is_first_k_block,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Matrix multiplication kernel tests
    //!
    //! Comprehensive tests for matrix multiplication kernels including accuracy,
    //! performance characteristics, and memory safety. Tests cover all kernel
    //! types and matrix size categories with validation against reference implementations.

    use super::super::config::MatmulConfig;
    use super::*;

    /// Reference implementation for testing
    fn reference_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                for ki in 0..k {
                    c[i * n + j] += a[i * k + ki] * b[ki * n + j];
                }
            }
        }
        c
    }

    /// Compare two matrices with tolerance
    fn matrices_equal(a: &[f32], b: &[f32], tolerance: f32) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|(&x, &y)| (x - y).abs() <= tolerance)
    }

    /// Test basic matrix multiplication kernel functionality
    ///
    /// Verifies that the kernel correctly computes matrix multiplication for small
    /// matrices and produces results matching the reference implementation.
    #[test]
    fn test_matrix_multiply_kernel_basic() {
        let m = 4;
        let n = 4;
        let k = 4;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i + 1) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i + 1) as f32).collect();
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            matrix_multiply_kernel(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                true,
            );
        }

        // Verify against reference implementation
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&c_data, &expected, 1e-6));
    }

    #[test]
    fn test_scalar_kernel_accuracy() {
        let m = 6;
        let n = 7;
        let k = 5;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7 + 1) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5 + 1) as f32).collect();
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            dispatch_ml_optimized_kernel_scalar(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                true,
            );
        }

        // Verify against reference implementation
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&c_data, &expected, 1e-6));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_kernel_accuracy() {
        if !is_x86_feature_detected!("avx2") {
            // Skip if AVX2 not available
            return;
        }

        let m = 8;
        let n = 16; // Multiple of 8 for optimal SIMD
        let k = 12;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 9 + 1) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 7 + 1) as f32).collect();
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            dispatch_ml_optimized_kernel_avx2(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                true,
            );
        }

        // Verify against reference implementation
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&c_data, &expected, 1e-6));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_vs_scalar_consistency() {
        if !is_x86_feature_detected!("avx2") {
            // Skip if AVX2 not available
            return;
        }

        let test_cases = vec![
            (4, 8, 4),    // SIMD-friendly
            (3, 9, 5),    // Mixed alignment
            (7, 15, 8),   // Non-SIMD dimensions
            (16, 32, 16), // Large SIMD-friendly
        ];

        for (m, n, k) in test_cases {
            let a_data: Vec<f32> = (0..m * k).map(|i| (i % 11 + 1) as f32).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| (i % 13 + 1) as f32).collect();

            let mut scalar_result = vec![0.0f32; m * n];
            let mut avx2_result = vec![0.0f32; m * n];

            unsafe {
                dispatch_ml_optimized_kernel_scalar(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    scalar_result.as_mut_ptr(),
                    m,
                    n,
                    k,
                    true,
                );

                dispatch_ml_optimized_kernel_avx2(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    avx2_result.as_mut_ptr(),
                    m,
                    n,
                    k,
                    true,
                );
            }

            // Both should produce identical results
            assert!(
                matrices_equal(&scalar_result, &avx2_result, 1e-6),
                "AVX2 and scalar results differ for {}x{}x{}",
                m,
                n,
                k
            );
        }
    }

    #[test]
    fn test_accumulation_mode() {
        let m = 4;
        let n = 4;
        let k = 4;

        let a_data: Vec<f32> = (1..=m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (1..=k * n).map(|i| i as f32).collect();
        let mut c_data = vec![2.0f32; m * n]; // Pre-initialized with 2.0

        unsafe {
            matrix_multiply_kernel(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                false, // Accumulation mode
            );
        }

        // Calculate expected result: reference + 2.0
        let expected_base = reference_matmul(&a_data, &b_data, m, n, k);
        let expected: Vec<f32> = expected_base.iter().map(|&x| x + 2.0).collect();

        assert!(matrices_equal(&c_data, &expected, 1e-6));
    }

    /// Test blocked matrix multiplication algorithm
    ///
    /// Verifies that the blocked matrix multiplication algorithm produces correct
    /// results for medium-sized matrices using the configuration system.
    #[test]
    fn test_matrix_multiply_blocked() {
        let m = 32;
        let n = 24;
        let k = 16;

        let a_data: Vec<f32> = (0..m * k).map(|i| (i % 7 + 1) as f32).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i % 5 + 1) as f32).collect();
        let mut c_data = vec![0.0f32; m * n];

        let config = MatmulConfig::for_dimensions(m, n, k);

        unsafe {
            matrix_multiply_blocked(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                &config,
            );
        }

        // Verify against reference implementation
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&c_data, &expected, 1e-5));
    }

    #[test]
    fn test_various_block_sizes() {
        let test_cases = vec![
            (8, 8, 8),    // Small
            (32, 32, 32), // Medium
            (64, 48, 40), // Non-square
            (17, 23, 19), // Prime dimensions
        ];

        for (m, n, k) in test_cases {
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i * 3) % 17 + 1) as f32).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i * 5) % 13 + 1) as f32).collect();
            let mut c_data = vec![0.0f32; m * n];

            let config = MatmulConfig::for_dimensions(m, n, k);

            unsafe {
                matrix_multiply_blocked(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    c_data.as_mut_ptr(),
                    m,
                    n,
                    k,
                    &config,
                );
            }

            let expected = reference_matmul(&a_data, &b_data, m, n, k);
            assert!(
                matrices_equal(&c_data, &expected, 1e-5),
                "Blocked matmul failed for {}x{}x{}",
                m,
                n,
                k
            );
        }
    }

    #[test]
    fn test_kernel_selection_dispatch() {
        let m = 16;
        let n = 16;
        let k = 16;

        let a_data: Vec<f32> = (1..=m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (1..=k * n).map(|i| i as f32).collect();
        let mut c_data = vec![0.0f32; m * n];

        unsafe {
            matrix_multiply_kernel(
                a_data.as_ptr(),
                b_data.as_ptr(),
                c_data.as_mut_ptr(),
                m,
                n,
                k,
                true,
            );
        }

        // Should use either AVX2 or scalar based on availability
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&c_data, &expected, 1e-6));
    }

    #[test]
    fn test_matmul_buffers() {
        let mut buffers = MatmulBuffers::new();
        assert_eq!(buffers.packed_b.len(), 1024);

        buffers.ensure_capacity(2048);
        assert_eq!(buffers.packed_b.len(), 2048);

        buffers.ensure_capacity(512);
        assert_eq!(buffers.packed_b.len(), 2048); // Should not shrink
    }

    #[test]
    fn test_ml_kernel_dispatch_sizes() {
        // Test that kernel dispatch selects appropriate kernels for ML-common sizes
        let test_cases = vec![
            // Small matrices: 16-64 range
            (16, 16, 16),
            (32, 32, 32),
            (64, 32, 48),
            // Medium-small matrices: 64-128 range
            (64, 64, 64),
            (96, 96, 96),
            (128, 64, 96),
            // Medium matrices: 128-256 range
            (128, 128, 128),
            (192, 192, 192),
            (256, 128, 192),
            // Large matrices: 256-512 range (smaller for test speed)
            (256, 256, 256),
            (384, 256, 192),
            (512, 128, 256),
        ];

        for (m, n, k) in test_cases {
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) + 1) as f32).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 5) + 1) as f32).collect();
            let mut result = vec![0.0f32; m * n];

            unsafe {
                matrix_multiply_kernel(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    result.as_mut_ptr(),
                    m,
                    n,
                    k,
                    true,
                );
            }

            // Verify against reference implementation
            let expected = reference_matmul(&a_data, &b_data, m, n, k);
            assert!(
                matrices_equal(&result, &expected, 1e-5),
                "ML kernel dispatch failed for {}x{}x{}",
                m,
                n,
                k
            );
        }
    }

    /// Test memory initialization safety with uninitialized memory
    ///
    /// Verifies that kernels handle uninitialized memory safely when used with
    /// `Tensor::new_uninitialized`, ensuring no uninitialized memory access.
    #[test]
    fn test_memory_initialization_safety() {
        // Test that the new kernel system handles uninitialized memory safely
        let m = 64;
        let n = 64;
        let k = 64;

        let a_data: Vec<f32> = (1..=m * k).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (1..=k * n).map(|i| i as f32).collect();

        // Simulate uninitialized memory (like Tensor::new_uninitialized would provide)
        // Use MaybeUninit to properly handle uninitialized memory
        use std::mem::MaybeUninit;
        let mut result_uninitialized: Vec<MaybeUninit<f32>> = Vec::with_capacity(m * n);
        unsafe {
            result_uninitialized.set_len(m * n);
        }

        // Cast to the expected type for our kernel (this simulates raw tensor data)
        let result_ptr = result_uninitialized.as_mut_ptr() as *mut f32;

        // Use our kernel with uninitialized memory (should be safe)
        unsafe {
            matrix_multiply_kernel(
                a_data.as_ptr(),
                b_data.as_ptr(),
                result_ptr,
                m,
                n,
                k,
                true, // First block - should initialize values safely
            );
        }

        // Convert back to initialized Vec<f32> for comparison
        let result_uninitialized: Vec<f32> = unsafe { std::mem::transmute(result_uninitialized) };

        // Compare with properly initialized result
        let mut result_initialized = vec![0.0f32; m * n];
        unsafe {
            matrix_multiply_kernel(
                a_data.as_ptr(),
                b_data.as_ptr(),
                result_initialized.as_mut_ptr(),
                m,
                n,
                k,
                true,
            );
        }

        // Results should be identical
        assert!(matrices_equal(
            &result_uninitialized,
            &result_initialized,
            1e-10
        ));
    }

    #[test]
    fn test_accumulation_safety() {
        // Test that accumulation mode works correctly
        let m = 32;
        let n = 32;
        let k = 64;

        let a_data: Vec<f32> = (1..=m * k).map(|i| (i % 10 + 1) as f32).collect();
        let b_data: Vec<f32> = (1..=k * n).map(|i| (i % 8 + 1) as f32).collect();

        // Use the blocked multiplication algorithm directly (like the real implementation)
        let config = MatmulConfig::for_dimensions(m, n, k);
        let mut result = vec![0.0f32; m * n];

        unsafe {
            matrix_multiply_blocked(
                a_data.as_ptr(),
                b_data.as_ptr(),
                result.as_mut_ptr(),
                m,
                n,
                k,
                &config,
            );
        }

        // Verify against reference implementation
        let expected = reference_matmul(&a_data, &b_data, m, n, k);
        assert!(matrices_equal(&result, &expected, 1e-5));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn test_avx2_vs_scalar_ml_sizes() {
        // Test AVX2 vs scalar consistency for ML-specific matrix sizes
        if !is_x86_feature_detected!("avx2") {
            return;
        }

        let ml_sizes = vec![
            (32, 32, 32),    // Small transformers
            (64, 64, 64),    // Small hidden layers
            (128, 128, 128), // Medium hidden layers
            (256, 128, 256), // Transformer attention
            (512, 256, 512), // Large hidden layers
        ];

        for (m, n, k) in ml_sizes {
            let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 11) + 1) as f32).collect();
            let b_data: Vec<f32> = (0..k * n).map(|i| ((i % 13) + 1) as f32).collect();

            let mut avx2_result = vec![0.0f32; m * n];
            let mut scalar_result = vec![0.0f32; m * n];

            unsafe {
                dispatch_ml_optimized_kernel_avx2(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    avx2_result.as_mut_ptr(),
                    m,
                    n,
                    k,
                    true,
                );

                dispatch_ml_optimized_kernel_scalar(
                    a_data.as_ptr(),
                    b_data.as_ptr(),
                    scalar_result.as_mut_ptr(),
                    m,
                    n,
                    k,
                    true,
                );
            }

            assert!(
                matrices_equal(&avx2_result, &scalar_result, 1e-6),
                "AVX2 vs scalar mismatch for ML size {}x{}x{}",
                m,
                n,
                k
            );
        }
    }
}
