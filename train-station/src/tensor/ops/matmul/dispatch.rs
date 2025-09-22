#[cfg(target_arch = "x86_64")]
use crate::tensor::core::memory::simd_alignment_bytes;
use crate::tensor::core::memory::{detect_runtime_simd, SimdLevel};
#[cfg(target_arch = "x86_64")]
use crate::tensor::ops::matmul::avx2_kernels as avx2;
use crate::tensor::ops::matmul::scalar_kernels::*;

/// Matrix multiplication operation types for kernel dispatch
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatMulOpType {
    /// 1D @ 1D: Dot product returning scalar
    Dot1D1D,
    /// 1D @ 2D: Vector-matrix multiplication (v^T * M)
    Vec1D2D,
    /// 2D @ 1D: Matrix-vector multiplication (M * v)
    Mat2D1D,
    /// 2D @ 2D: Standard matrix multiplication
    Mat2D2D,
    /// ND @ ND: Batched matrix multiplication on last two dimensions
    BatchedND,
}

/// Matrix size classification for kernel selection
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixSizeClass {
    /// Small matrices (≤1K elements): Direct computation with minimal overhead
    Small,
    /// Medium matrices (1K-64K elements): Cache-optimized blocking for L1/L2 cache
    Medium,
    /// Large matrices (≥64K elements): Memory bandwidth optimized with hierarchical blocking
    Large,
}

/// Cached matrix multiplication kernels with intelligent dispatch
pub struct MatMulKernels {
    #[allow(dead_code)]
    pub simd_level: SimdLevel,
    #[allow(dead_code)]
    pub alignment: usize,

    // 1D @ 1D: Dot product kernels
    pub dot_1d_aligned: unsafe fn(*const f32, *const f32, usize) -> f32,
    pub dot_1d_unaligned: unsafe fn(*const f32, *const f32, usize) -> f32,

    // 1D @ 2D: Vector-matrix kernels (v^T * M)
    pub vec_mat_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize),
    pub vec_mat_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize),

    // 2D @ 1D: Matrix-vector kernels (M * v)
    pub mat_vec_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize),
    pub mat_vec_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize),

    // 2D @ 2D: Matrix-matrix kernels
    pub mat_mat_small_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),
    pub mat_mat_small_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),
    pub mat_mat_medium_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),
    pub mat_mat_medium_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),
    pub mat_mat_large_aligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),
    pub mat_mat_large_unaligned: unsafe fn(*const f32, *const f32, *mut f32, usize, usize, usize),

    // Size thresholds for kernel selection
    pub small_threshold: usize,  // Elements threshold for small matrices
    pub medium_threshold: usize, // Elements threshold for medium matrices
    pub min_aligned_size: usize, // Minimum size for aligned kernels
}

impl MatMulKernels {
    /// Get cached matmul kernels - single initialization, maximum performance
    #[inline]
    pub fn get_cached_kernels() -> &'static MatMulKernels {
        use std::sync::OnceLock;

        static CACHED_KERNELS: OnceLock<MatMulKernels> = OnceLock::new();

        CACHED_KERNELS.get_or_init(|| {
            let simd_level = detect_runtime_simd();
            #[cfg(target_arch = "x86_64")]
            let alignment = simd_alignment_bytes(simd_level);

            #[cfg(target_arch = "x86_64")]
            {
                match simd_level {
                    SimdLevel::Avx512 => MatMulKernels {
                        simd_level,
                        alignment,
                        // AVX512 kernels (16-wide SIMD)
                        dot_1d_aligned: Self::dot_1d_avx512_aligned,
                        dot_1d_unaligned: Self::dot_1d_avx512_unaligned,
                        vec_mat_aligned: Self::vec_mat_avx512_aligned,
                        vec_mat_unaligned: Self::vec_mat_avx512_unaligned,
                        mat_vec_aligned: Self::mat_vec_avx512_aligned,
                        mat_vec_unaligned: Self::mat_vec_avx512_unaligned,
                        mat_mat_small_aligned: Self::mat_mat_small_avx512_aligned,
                        mat_mat_small_unaligned: Self::mat_mat_small_avx512_unaligned,
                        mat_mat_medium_aligned: Self::mat_mat_medium_avx512_aligned,
                        mat_mat_medium_unaligned: Self::mat_mat_medium_avx512_unaligned,
                        mat_mat_large_aligned: Self::mat_mat_large_avx512_aligned,
                        mat_mat_large_unaligned: Self::mat_mat_large_avx512_unaligned,
                        small_threshold: 1024,   // 1K elements
                        medium_threshold: 65536, // 64K elements
                        min_aligned_size: 16,    // AVX512 vector width
                    },
                    SimdLevel::Avx2 => MatMulKernels {
                        simd_level,
                        alignment,
                        // AVX2 kernels (8-wide SIMD)
                        dot_1d_aligned: Self::dot_1d_avx2_aligned,
                        dot_1d_unaligned: Self::dot_1d_avx2_unaligned,
                        vec_mat_aligned: Self::vec_mat_avx2_aligned,
                        vec_mat_unaligned: Self::vec_mat_avx2_unaligned,
                        mat_vec_aligned: Self::mat_vec_avx2_aligned,
                        mat_vec_unaligned: Self::mat_vec_avx2_unaligned,
                        mat_mat_small_aligned: Self::mat_mat_small_avx2_aligned,
                        mat_mat_small_unaligned: Self::mat_mat_small_avx2_unaligned,
                        mat_mat_medium_aligned: Self::mat_mat_medium_avx2_aligned,
                        mat_mat_medium_unaligned: Self::mat_mat_medium_avx2_unaligned,
                        mat_mat_large_aligned: Self::mat_mat_large_avx2_aligned,
                        mat_mat_large_unaligned: Self::mat_mat_large_avx2_unaligned,
                        small_threshold: 1024,
                        medium_threshold: 65536,
                        min_aligned_size: 8, // AVX2 vector width
                    },
                    SimdLevel::Sse2 => MatMulKernels {
                        simd_level,
                        alignment,
                        // SSE2 kernels (4-wide SIMD)
                        dot_1d_aligned: Self::dot_1d_sse_aligned,
                        dot_1d_unaligned: Self::dot_1d_sse_unaligned,
                        vec_mat_aligned: Self::vec_mat_sse_aligned,
                        vec_mat_unaligned: Self::vec_mat_sse_unaligned,
                        mat_vec_aligned: Self::mat_vec_sse_aligned,
                        mat_vec_unaligned: Self::mat_vec_sse_unaligned,
                        mat_mat_small_aligned: Self::mat_mat_small_sse_aligned,
                        mat_mat_small_unaligned: Self::mat_mat_small_sse_unaligned,
                        mat_mat_medium_aligned: Self::mat_mat_medium_sse_aligned,
                        mat_mat_medium_unaligned: Self::mat_mat_medium_sse_unaligned,
                        mat_mat_large_aligned: Self::mat_mat_large_sse_aligned,
                        mat_mat_large_unaligned: Self::mat_mat_large_sse_unaligned,
                        small_threshold: 1024,
                        medium_threshold: 65536,
                        min_aligned_size: 4, // SSE2 vector width
                    },
                    SimdLevel::Scalar => MatMulKernels {
                        simd_level,
                        alignment: 4, // f32 alignment
                        // Scalar fallback kernels
                        dot_1d_aligned: Self::dot_1d_scalar,
                        dot_1d_unaligned: Self::dot_1d_scalar,
                        vec_mat_aligned: Self::vec_mat_scalar,
                        vec_mat_unaligned: Self::vec_mat_scalar,
                        mat_vec_aligned: Self::mat_vec_scalar,
                        mat_vec_unaligned: Self::mat_vec_scalar,
                        mat_mat_small_aligned: Self::mat_mat_scalar,
                        mat_mat_small_unaligned: Self::mat_mat_scalar,
                        mat_mat_medium_aligned: Self::mat_mat_scalar,
                        mat_mat_medium_unaligned: Self::mat_mat_scalar,
                        mat_mat_large_aligned: Self::mat_mat_scalar,
                        mat_mat_large_unaligned: Self::mat_mat_scalar,
                        small_threshold: 1024,
                        medium_threshold: 65536,
                        min_aligned_size: 1,
                    },
                }
            }

            #[cfg(not(target_arch = "x86_64"))]
            {
                MatMulKernels {
                    simd_level,
                    alignment: 4, // f32 alignment
                    // Non-x86 scalar fallback
                    dot_1d_aligned: Self::dot_1d_scalar,
                    dot_1d_unaligned: Self::dot_1d_scalar,
                    vec_mat_aligned: Self::vec_mat_scalar,
                    vec_mat_unaligned: Self::vec_mat_scalar,
                    mat_vec_aligned: Self::mat_vec_scalar,
                    mat_vec_unaligned: Self::mat_vec_scalar,
                    mat_mat_small_aligned: Self::mat_mat_scalar,
                    mat_mat_small_unaligned: Self::mat_mat_scalar,
                    mat_mat_medium_aligned: Self::mat_mat_scalar,
                    mat_mat_medium_unaligned: Self::mat_mat_scalar,
                    mat_mat_large_aligned: Self::mat_mat_scalar,
                    mat_mat_large_unaligned: Self::mat_mat_scalar,
                    small_threshold: 1024,
                    medium_threshold: 65536,
                    min_aligned_size: 1,
                }
            }
        })
    }

    /// Classify matrix operation type based on tensor shapes
    #[inline]
    pub fn classify_operation(left_shape: &[usize], right_shape: &[usize]) -> MatMulOpType {
        match (left_shape.len(), right_shape.len()) {
            (1, 1) => MatMulOpType::Dot1D1D,
            (1, 2) => MatMulOpType::Vec1D2D,
            (2, 1) => MatMulOpType::Mat2D1D,
            (2, 2) => MatMulOpType::Mat2D2D,
            _ => MatMulOpType::BatchedND,
        }
    }

    /// Classify matrix size for kernel selection
    #[inline]
    pub fn classify_matrix_size(&self, total_elements: usize) -> MatrixSizeClass {
        if total_elements <= self.small_threshold {
            MatrixSizeClass::Small
        } else if total_elements <= self.medium_threshold {
            MatrixSizeClass::Medium
        } else {
            MatrixSizeClass::Large
        }
    }

    /// Check alignment for SIMD operations
    #[inline]
    #[allow(dead_code)]
    pub fn check_alignment_for_simd(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        alignment: usize,
    ) -> bool {
        let a_aligned = (a_ptr as usize).is_multiple_of(alignment);
        let b_aligned = (b_ptr as usize).is_multiple_of(alignment);
        let c_aligned = (c_ptr as usize).is_multiple_of(alignment);
        a_aligned && b_aligned && c_aligned
    }

    /// Check actual alignment of pointers (conservative approach)
    #[inline]
    pub fn check_actual_alignment(
        &self,
        _a_ptr: *const f32,
        _b_ptr: *const f32,
        _c_ptr: *mut f32,
    ) -> bool {
        let alignment = self.alignment;
        let a_aligned = (_a_ptr as usize).is_multiple_of(alignment);
        let b_aligned = (_b_ptr as usize).is_multiple_of(alignment);
        // c_ptr may be null for dot kernels; treat null as aligned for purposes of selecting kernels
        let c_aligned = _c_ptr.is_null() || (_c_ptr as usize).is_multiple_of(alignment);
        a_aligned && b_aligned && c_aligned
    }

    /// Dispatch dot product operation (1D @ 1D)
    #[inline]
    pub unsafe fn dispatch_dot_1d(&self, a_ptr: *const f32, b_ptr: *const f32, size: usize) -> f32 {
        if size >= self.min_aligned_size
            && self.check_actual_alignment(a_ptr, b_ptr, std::ptr::null_mut())
        {
            (self.dot_1d_aligned)(a_ptr, b_ptr, size)
        } else {
            (self.dot_1d_unaligned)(a_ptr, b_ptr, size)
        }
    }

    /// Dispatch dot product operation with strides (1D @ 1D)
    #[inline]
    pub unsafe fn dispatch_dot_1d_strided(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        size: usize,
        a_stride: usize,
        b_stride: usize,
    ) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if matches!(self.simd_level, SimdLevel::Avx2) {
                // Use AVX2 strided dot with gathers; no alignment requirement
                return avx2::dot_1d_avx2_strided(a_ptr, b_ptr, size, a_stride, b_stride);
            }
        }
        // Fallback to scalar for non-x86
        matmul_scalar_1d_1d(a_ptr, b_ptr, size, a_stride, b_stride)
    }

    /// Dispatch vector-matrix operation (1D @ 2D)
    #[inline]
    pub unsafe fn dispatch_vec_mat(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        k: usize,
        n: usize,
    ) {
        if k >= self.min_aligned_size && self.check_actual_alignment(a_ptr, b_ptr, c_ptr) {
            (self.vec_mat_aligned)(a_ptr, b_ptr, c_ptr, k, n);
        } else {
            (self.vec_mat_unaligned)(a_ptr, b_ptr, c_ptr, k, n);
        }
    }

    /// Dispatch vector-matrix operation with strides (1D @ 2D)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_vec_mat_strided(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        k: usize,
        n: usize,
        a_stride: usize,
        b_row_stride: usize,
        b_col_stride: usize,
        c_stride: usize,
    ) {
        // Try AVX2 fast-path when strides correspond to contiguous layout
        // Contiguous layout check for row-major [K] @ [K,N] -> [N]
        // a_stride==1, b_row_stride==n, b_col_stride==1, c_stride==1
        if a_stride == 1 && b_row_stride == n && b_col_stride == 1 && c_stride == 1 {
            if k >= self.min_aligned_size && self.check_actual_alignment(a_ptr, b_ptr, c_ptr) {
                (self.vec_mat_aligned)(a_ptr, b_ptr, c_ptr, k, n)
            } else {
                (self.vec_mat_unaligned)(a_ptr, b_ptr, c_ptr, k, n)
            }
        } else {
            // Arbitrary strides: try AVX2 packed strided path if available
            #[cfg(target_arch = "x86_64")]
            {
                if matches!(self.simd_level, SimdLevel::Avx2) {
                    let use_aligned = k >= self.min_aligned_size
                        && self.check_actual_alignment(a_ptr, b_ptr, c_ptr);
                    // Treat as 2D: [1,K] @ [K,N] -> [1,N]
                    let a_row_stride2d = 0usize; // single row
                    let a_col_stride2d = a_stride;
                    let c_row_stride2d = 0usize; // single row
                    let c_col_stride2d = c_stride;
                    if use_aligned {
                        return avx2::mat_mat_strided_avx2_aligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            1,
                            k,
                            n,
                            a_row_stride2d,
                            a_col_stride2d,
                            b_row_stride,
                            b_col_stride,
                            c_row_stride2d,
                            c_col_stride2d,
                        );
                    } else {
                        return avx2::mat_mat_strided_avx2_unaligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            1,
                            k,
                            n,
                            a_row_stride2d,
                            a_col_stride2d,
                            b_row_stride,
                            b_col_stride,
                            c_row_stride2d,
                            c_col_stride2d,
                        );
                    }
                }
            }

            // Fallback to scalar for non-x86 or non-AVX2
            matmul_scalar_1d_2d(
                a_ptr,
                b_ptr,
                c_ptr,
                k,
                n,
                a_stride,
                b_row_stride,
                b_col_stride,
                c_stride,
            )
        }
    }

    /// Dispatch matrix-vector operation (2D @ 1D)
    #[inline]
    pub unsafe fn dispatch_mat_vec(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        m: usize,
        k: usize,
    ) {
        if k >= self.min_aligned_size && self.check_actual_alignment(a_ptr, b_ptr, c_ptr) {
            (self.mat_vec_aligned)(a_ptr, b_ptr, c_ptr, m, k);
        } else {
            (self.mat_vec_unaligned)(a_ptr, b_ptr, c_ptr, m, k);
        }
    }

    /// Dispatch matrix-vector operation with strides (2D @ 1D)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_mat_vec_strided(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        m: usize,
        k: usize,
        a_row_stride: usize,
        a_col_stride: usize,
        b_stride: usize,
        c_stride: usize,
    ) {
        // Try AVX2 fast-path when strides correspond to contiguous layout
        // Contiguous layout check for row-major [M,K] @ [K] -> [M]
        // a_row_stride==k, a_col_stride==1, b_stride==1, c_stride==1
        if a_row_stride == k && a_col_stride == 1 && b_stride == 1 && c_stride == 1 {
            if k >= self.min_aligned_size && self.check_actual_alignment(a_ptr, b_ptr, c_ptr) {
                (self.mat_vec_aligned)(a_ptr, b_ptr, c_ptr, m, k)
            } else {
                (self.mat_vec_unaligned)(a_ptr, b_ptr, c_ptr, m, k)
            }
        } else {
            // Arbitrary strides: try AVX2 packed strided path if available
            #[cfg(target_arch = "x86_64")]
            {
                if matches!(self.simd_level, SimdLevel::Avx2) {
                    let use_aligned = k >= self.min_aligned_size
                        && self.check_actual_alignment(a_ptr, b_ptr, c_ptr);
                    // Treat as 2D: [M,K] @ [K,1] -> [M,1]
                    let b_row_stride2d = b_stride;
                    let b_col_stride2d = 0usize; // single column
                    let c_row_stride2d = c_stride;
                    let c_col_stride2d = 0usize; // single column
                    if use_aligned {
                        return avx2::mat_mat_strided_avx2_aligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            m,
                            k,
                            1,
                            a_row_stride,
                            a_col_stride,
                            b_row_stride2d,
                            b_col_stride2d,
                            c_row_stride2d,
                            c_col_stride2d,
                        );
                    } else {
                        return avx2::mat_mat_strided_avx2_unaligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            m,
                            k,
                            1,
                            a_row_stride,
                            a_col_stride,
                            b_row_stride2d,
                            b_col_stride2d,
                            c_row_stride2d,
                            c_col_stride2d,
                        );
                    }
                }
            }

            // Fallback to scalar for non-x86 or non-AVX2
            matmul_scalar_2d_1d(
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                k,
                a_row_stride,
                a_col_stride,
                b_stride,
                c_stride,
            )
        }
    }

    /// Dispatch matrix-matrix operation (2D @ 2D)
    #[inline]
    pub unsafe fn dispatch_mat_mat(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Enable AVX2 for all K; numerical parity validated externally

        let total_elements = m * k + k * n + m * n;
        let size_class = self.classify_matrix_size(total_elements);
        let use_aligned =
            k >= self.min_aligned_size && self.check_actual_alignment(a_ptr, b_ptr, c_ptr);

        match (size_class, use_aligned) {
            (MatrixSizeClass::Small, true) => {
                (self.mat_mat_small_aligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
            (MatrixSizeClass::Small, false) => {
                (self.mat_mat_small_unaligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
            (MatrixSizeClass::Medium, true) => {
                (self.mat_mat_medium_aligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
            (MatrixSizeClass::Medium, false) => {
                (self.mat_mat_medium_unaligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
            (MatrixSizeClass::Large, true) => {
                (self.mat_mat_large_aligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
            (MatrixSizeClass::Large, false) => {
                (self.mat_mat_large_unaligned)(a_ptr, b_ptr, c_ptr, m, k, n)
            }
        }
    }

    /// Dispatch matrix-matrix operation with strides (2D @ 2D)
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn dispatch_mat_mat_strided(
        &self,
        a_ptr: *const f32,
        b_ptr: *const f32,
        c_ptr: *mut f32,
        m: usize,
        k: usize,
        n: usize,
        a_row_stride: usize,
        a_col_stride: usize,
        b_row_stride: usize,
        b_col_stride: usize,
        c_row_stride: usize,
        c_col_stride: usize,
    ) {
        // Try AVX2 fast-path when strides correspond to contiguous layout
        // Contiguous layout check for row-major [M,K] @ [K,N] -> [M,N]
        // a_row_stride==k, a_col_stride==1, b_row_stride==n, b_col_stride==1,
        // c_row_stride==n, c_col_stride==1
        if a_row_stride == k
            && a_col_stride == 1
            && b_row_stride == n
            && b_col_stride == 1
            && c_row_stride == n
            && c_col_stride == 1
        {
            self.dispatch_mat_mat(a_ptr, b_ptr, c_ptr, m, k, n)
        } else {
            // Arbitrary strides: try AVX2 packed strided path if available
            #[cfg(target_arch = "x86_64")]
            {
                if matches!(self.simd_level, SimdLevel::Avx2) {
                    let use_aligned = k >= self.min_aligned_size
                        && self.check_actual_alignment(a_ptr, b_ptr, c_ptr);
                    if use_aligned {
                        return avx2::mat_mat_strided_avx2_aligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            m,
                            k,
                            n,
                            a_row_stride,
                            a_col_stride,
                            b_row_stride,
                            b_col_stride,
                            c_row_stride,
                            c_col_stride,
                        );
                    } else {
                        return avx2::mat_mat_strided_avx2_unaligned(
                            a_ptr,
                            b_ptr,
                            c_ptr,
                            m,
                            k,
                            n,
                            a_row_stride,
                            a_col_stride,
                            b_row_stride,
                            b_col_stride,
                            c_row_stride,
                            c_col_stride,
                        );
                    }
                }
            }

            // Fallback to scalar for non-x86 or non-AVX2
            matmul_scalar_2d_2d(
                a_ptr,
                b_ptr,
                c_ptr,
                m,
                k,
                n,
                a_row_stride,
                a_col_stride,
                b_row_stride,
                b_col_stride,
                c_row_stride,
                c_col_stride,
            )
        }
    }

    // Placeholder kernel implementations - will be implemented in subsequent phases

    // Scalar fallback kernels
    #[inline]
    unsafe fn dot_1d_scalar(a: *const f32, b: *const f32, size: usize) -> f32 {
        // Use optimized scalar kernel with contiguous strides
        matmul_scalar_1d_1d(a, b, size, 1, 1)
    }

    #[inline]
    unsafe fn vec_mat_scalar(a: *const f32, b: *const f32, c: *mut f32, k: usize, n: usize) {
        // Use optimized scalar kernel with contiguous strides
        // For contiguous tensors: a_stride=1, b_row_stride=n, b_col_stride=1, c_stride=1
        matmul_scalar_1d_2d(a, b, c, k, n, 1, n, 1, 1)
    }

    #[inline]
    unsafe fn mat_vec_scalar(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize) {
        // Use optimized scalar kernel with contiguous strides
        // For contiguous tensors: a_row_stride=k, a_col_stride=1, b_stride=1, c_stride=1
        matmul_scalar_2d_1d(a, b, c, m, k, k, 1, 1, 1)
    }

    #[inline]
    unsafe fn mat_mat_scalar(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        // Use optimized scalar kernel with contiguous strides
        // For contiguous tensors: a_row_stride=k, a_col_stride=1, b_row_stride=n, b_col_stride=1, c_row_stride=n, c_col_stride=1
        matmul_scalar_2d_2d(a, b, c, m, k, n, k, 1, n, 1, n, 1)
    }

    // SIMD kernel placeholders - to be implemented in subsequent phases

    // AVX512 kernels
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_1d_avx512_aligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        Self::dot_1d_scalar(a, b, size) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn dot_1d_avx512_unaligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        Self::dot_1d_scalar(a, b, size) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn vec_mat_avx512_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        k: usize,
        n: usize,
    ) {
        Self::vec_mat_scalar(a, b, c, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn vec_mat_avx512_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        k: usize,
        n: usize,
    ) {
        Self::vec_mat_scalar(a, b, c, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_vec_avx512_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
    ) {
        Self::mat_vec_scalar(a, b, c, m, k) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_vec_avx512_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
    ) {
        Self::mat_vec_scalar(a, b, c, m, k) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_small_avx512_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_small_avx512_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_medium_avx512_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_medium_avx512_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_large_avx512_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx512f")]
    unsafe fn mat_mat_large_avx512_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    // AVX2 kernels (similar structure - placeholders for now)
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_1d_avx2_aligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        avx2::dot_1d_avx2_aligned(a, b, size)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn dot_1d_avx2_unaligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        avx2::dot_1d_avx2_unaligned(a, b, size)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vec_mat_avx2_aligned(a: *const f32, b: *const f32, c: *mut f32, k: usize, n: usize) {
        avx2::vec_mat_avx2_aligned(a, b, c, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn vec_mat_avx2_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        k: usize,
        n: usize,
    ) {
        avx2::vec_mat_avx2_unaligned(a, b, c, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_vec_avx2_aligned(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize) {
        avx2::mat_vec_avx2_aligned(a, b, c, m, k)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_vec_avx2_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
    ) {
        avx2::mat_vec_avx2_unaligned(a, b, c, m, k)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_small_avx2_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_small_avx2_aligned(a, b, c, m, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_small_avx2_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_small_avx2_unaligned(a, b, c, m, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_medium_avx2_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_medium_avx2_aligned(a, b, c, m, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_medium_avx2_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_medium_avx2_unaligned(a, b, c, m, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_large_avx2_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_large_avx2_aligned(a, b, c, m, k, n)
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn mat_mat_large_avx2_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        avx2::mat_mat_large_avx2_unaligned(a, b, c, m, k, n)
    }

    // SSE2 kernels (similar structure - placeholders for now)
    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_1d_sse_aligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        Self::dot_1d_scalar(a, b, size) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_1d_sse_unaligned(a: *const f32, b: *const f32, size: usize) -> f32 {
        Self::dot_1d_scalar(a, b, size) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn vec_mat_sse_aligned(a: *const f32, b: *const f32, c: *mut f32, k: usize, n: usize) {
        Self::vec_mat_scalar(a, b, c, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn vec_mat_sse_unaligned(a: *const f32, b: *const f32, c: *mut f32, k: usize, n: usize) {
        Self::vec_mat_scalar(a, b, c, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_vec_sse_aligned(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize) {
        Self::mat_vec_scalar(a, b, c, m, k) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_vec_sse_unaligned(a: *const f32, b: *const f32, c: *mut f32, m: usize, k: usize) {
        Self::mat_vec_scalar(a, b, c, m, k) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_small_sse_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_small_sse_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_medium_sse_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_medium_sse_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_large_sse_aligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn mat_mat_large_sse_unaligned(
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        m: usize,
        k: usize,
        n: usize,
    ) {
        Self::mat_mat_scalar(a, b, c, m, k, n) // Placeholder
    }
}
