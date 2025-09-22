//! AVX2-optimized kernels for matmul operations.
#![allow(clippy::too_many_arguments)]

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use crate::tensor::ops::matmul::pack_n_cache::with_panel_cache;
use crate::tensor::ops::matmul::pack_n_cache::BlockingParams;
use crate::tensor::ops::matmul::pack_n_cache::PackedPanelA;
// Packed panels are referenced via cache accessors; explicit types not needed here

// Prefetch distance for kc loop (in iterations)
// Tuned dynamically per micro-kernel based on kc to improve L1/L2 effectiveness
#[inline(always)]
fn prefetch_distance_kc(kc: usize) -> usize {
    if kc >= 192 {
        4
    } else if kc >= 96 {
        2
    } else if kc >= 48 {
        1
    } else {
        0
    }
}

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(lo, hi);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(shuf, sums);
    let sums2 = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(sums2)
}

/// AVX2 unaligned dot product (contiguous stride=1)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_1d_avx2_unaligned(a_ptr: *const f32, b_ptr: *const f32, size: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    // Process 16 at a time (2x ymm)
    let main16 = size / 16 * 16;
    while i < main16 {
        let a0 = _mm256_loadu_ps(a_ptr.add(i));
        let b0 = _mm256_loadu_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = _mm256_loadu_ps(a_ptr.add(i + 8));
        let b1 = _mm256_loadu_ps(b_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }

    // Process remaining 8
    let mut acc_tail = _mm256_setzero_ps();
    if i + 8 <= size {
        let a = _mm256_loadu_ps(a_ptr.add(i));
        let b = _mm256_loadu_ps(b_ptr.add(i));
        acc_tail = _mm256_fmadd_ps(a, b, acc_tail);
        i += 8;
    }

    let mut sum = hsum256_ps(acc0) + hsum256_ps(acc1) + hsum256_ps(acc_tail);

    // Scalar tail
    while i < size {
        sum += *a_ptr.add(i) * *b_ptr.add(i);
        i += 1;
    }
    sum
}

/// AVX2 dot with arbitrary strides via gathers
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_1d_avx2_strided(
    a_ptr: *const f32,
    b_ptr: *const f32,
    size: usize,
    a_stride: usize,
    b_stride: usize,
) -> f32 {
    // Prefer tiled contiguous path for irregular strides to avoid expensive gathers
    const TILE: usize = 256; // 1KB per array per tile
    let irregular = a_stride != 1 || b_stride != 1;
    if irregular && size >= TILE * 2 {
        #[repr(align(32))]
        struct Tile([f32; TILE]);
        let mut total = 0.0f32;
        let mut i = 0usize;
        while i + TILE <= size {
            let mut ta = Tile([0.0; TILE]);
            let mut tb = Tile([0.0; TILE]);
            let tap = ta.0.as_mut_ptr();
            let tbp = tb.0.as_mut_ptr();
            let mut t = 0usize;
            while t < TILE {
                #[allow(unused_unsafe)]
                unsafe {
                    if t + 16 < TILE {
                        _mm_prefetch(a_ptr.add((i + t + 16) * a_stride) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(b_ptr.add((i + t + 16) * b_stride) as *const i8, _MM_HINT_T0);
                    }
                }
                *tap.add(t) = *a_ptr.add((i + t) * a_stride);
                *tbp.add(t) = *b_ptr.add((i + t) * b_stride);
                t += 1;
            }
            total += dot_1d_avx2_unaligned(tap, tbp, TILE);
            i += TILE;
        }
        while i < size {
            total += *a_ptr.add(i * a_stride) * *b_ptr.add(i * b_stride);
            i += 1;
        }
        return total;
    }

    // Default: AVX2 gather path
    let mut i = 0usize;
    let mut acc = _mm256_setzero_ps();

    let idx_base = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    let stride_a = _mm256_set1_epi32(a_stride as i32);
    let stride_b = _mm256_set1_epi32(b_stride as i32);
    let main = size / 8 * 8;
    while i < main {
        let offset_a = _mm256_set1_epi32((i as i32) * (a_stride as i32));
        let offset_b = _mm256_set1_epi32((i as i32) * (b_stride as i32));
        let idx_a = _mm256_add_epi32(_mm256_mullo_epi32(idx_base, stride_a), offset_a);
        let idx_b = _mm256_add_epi32(_mm256_mullo_epi32(idx_base, stride_b), offset_b);
        let va = _mm256_i32gather_ps(a_ptr, idx_a, 4);
        let vb = _mm256_i32gather_ps(b_ptr, idx_b, 4);
        acc = _mm256_fmadd_ps(va, vb, acc);
        i += 8;
    }
    let mut sum = hsum256_ps(acc);
    while i < size {
        sum += *a_ptr.add(i * a_stride) * *b_ptr.add(i * b_stride);
        i += 1;
    }
    sum
}

/// AVX2 aligned dot product (currently defers to unaligned to stay safe)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_1d_avx2_aligned(a_ptr: *const f32, b_ptr: *const f32, size: usize) -> f32 {
    let mut i = 0usize;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let main16 = size / 16 * 16;
    while i < main16 {
        let a0 = _mm256_load_ps(a_ptr.add(i));
        let b0 = _mm256_load_ps(b_ptr.add(i));
        acc0 = _mm256_fmadd_ps(a0, b0, acc0);

        let a1 = _mm256_load_ps(a_ptr.add(i + 8));
        let b1 = _mm256_load_ps(b_ptr.add(i + 8));
        acc1 = _mm256_fmadd_ps(a1, b1, acc1);
        i += 16;
    }

    let mut acc_tail = _mm256_setzero_ps();
    if i + 8 <= size {
        let a = _mm256_load_ps(a_ptr.add(i));
        let b = _mm256_load_ps(b_ptr.add(i));
        acc_tail = _mm256_fmadd_ps(a, b, acc_tail);
        i += 8;
    }

    let mut sum = hsum256_ps(acc0) + hsum256_ps(acc1) + hsum256_ps(acc_tail);
    while i < size {
        sum += *a_ptr.add(i) * *b_ptr.add(i);
        i += 1;
    }
    sum
}

/// v^T (K) @ M (K,N) -> out (N)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_mat_avx2_unaligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
) {
    let mut j = 0usize;
    // Process 16 columns at a time
    while j + 16 <= n {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        let mut kk = 0usize;
        while kk + 8 <= k {
            let a8 = _mm256_loadu_ps(a_ptr.add(kk));
            let a_lane: [f32; 8] = core::mem::transmute(a8);
            for (t, &a_lane_t) in a_lane.iter().enumerate() {
                let b_row = b_ptr.add((kk + t) * n + j);
                let vb0 = _mm256_loadu_ps(b_row);
                let vb1 = _mm256_loadu_ps(b_row.add(8));
                let as_broadcast = _mm256_set1_ps(a_lane_t);
                acc0 = _mm256_fmadd_ps(as_broadcast, vb0, acc0);
                acc1 = _mm256_fmadd_ps(as_broadcast, vb1, acc1);
            }
            kk += 8;
        }

        while kk < k {
            let a_scalar = *a_ptr.add(kk);
            let as_broadcast = _mm256_set1_ps(a_scalar);
            let b_row = b_ptr.add(kk * n + j);
            let vb0 = _mm256_loadu_ps(b_row);
            let vb1 = _mm256_loadu_ps(b_row.add(8));
            acc0 = _mm256_fmadd_ps(as_broadcast, vb0, acc0);
            acc1 = _mm256_fmadd_ps(as_broadcast, vb1, acc1);
            kk += 1;
        }

        _mm256_storeu_ps(c_ptr.add(j), acc0);
        _mm256_storeu_ps(c_ptr.add(j + 8), acc1);
        j += 16;
    }

    // Process 8 columns
    if j + 8 <= n {
        let mut acc = _mm256_setzero_ps();
        let mut kk = 0usize;
        while kk < k {
            let as_broadcast = _mm256_set1_ps(*a_ptr.add(kk));
            let b_row = b_ptr.add(kk * n + j);
            let vb = _mm256_loadu_ps(b_row);
            acc = _mm256_fmadd_ps(as_broadcast, vb, acc);
            kk += 1;
        }
        _mm256_storeu_ps(c_ptr.add(j), acc);
        j += 8;
    }

    // Tail columns
    while j < n {
        let mut sum = 0.0f32;
        for kk in 0..k {
            sum += *a_ptr.add(kk) * *b_ptr.add(kk * n + j);
        }
        *c_ptr.add(j) = sum;
        j += 1;
    }
}

/// AVX2 aligned v^T@M (delegates to unaligned for safety)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn vec_mat_avx2_aligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    k: usize,
    n: usize,
) {
    let mut j = 0usize;
    while j + 16 <= n {
        let mut acc0 = _mm256_setzero_ps();
        let mut acc1 = _mm256_setzero_ps();

        // Check alignment of this output block and corresponding B columns
        let c_block_aligned = (c_ptr.add(j) as usize).is_multiple_of(32);
        let mut kk = 0usize;
        // Peel until a_ptr is aligned (for aligned loads on A)
        while kk < k && !(a_ptr.add(kk) as usize).is_multiple_of(32) {
            let as_broadcast = _mm256_set1_ps(*a_ptr.add(kk));
            let b_row = b_ptr.add(kk * n + j);
            let vb0 = _mm256_loadu_ps(b_row);
            let vb1 = _mm256_loadu_ps(b_row.add(8));
            acc0 = _mm256_fmadd_ps(as_broadcast, vb0, acc0);
            acc1 = _mm256_fmadd_ps(as_broadcast, vb1, acc1);
            kk += 1;
        }

        while kk + 8 <= k {
            let a8 = _mm256_load_ps(a_ptr.add(kk));
            let a_lane: [f32; 8] = core::mem::transmute(a8);
            for (t, &a_lane_t) in a_lane.iter().enumerate() {
                let b_row = b_ptr.add((kk + t) * n + j);
                let vb0 = if (b_row as usize).is_multiple_of(32) {
                    _mm256_load_ps(b_row)
                } else {
                    _mm256_loadu_ps(b_row)
                };
                let vb1 = if (b_row.add(8) as usize).is_multiple_of(32) {
                    _mm256_load_ps(b_row.add(8))
                } else {
                    _mm256_loadu_ps(b_row.add(8))
                };
                let as_broadcast = _mm256_set1_ps(a_lane_t);
                acc0 = _mm256_fmadd_ps(as_broadcast, vb0, acc0);
                acc1 = _mm256_fmadd_ps(as_broadcast, vb1, acc1);
            }
            kk += 8;
        }

        while kk < k {
            let a_scalar = *a_ptr.add(kk);
            let as_broadcast = _mm256_set1_ps(a_scalar);
            let b_row = b_ptr.add(kk * n + j);
            let vb0 = if (b_row as usize).is_multiple_of(32) {
                _mm256_load_ps(b_row)
            } else {
                _mm256_loadu_ps(b_row)
            };
            let vb1 = if (b_row.add(8) as usize).is_multiple_of(32) {
                _mm256_load_ps(b_row.add(8))
            } else {
                _mm256_loadu_ps(b_row.add(8))
            };
            acc0 = _mm256_fmadd_ps(as_broadcast, vb0, acc0);
            acc1 = _mm256_fmadd_ps(as_broadcast, vb1, acc1);
            kk += 1;
        }

        if c_block_aligned {
            _mm256_store_ps(c_ptr.add(j), acc0);
            _mm256_store_ps(c_ptr.add(j + 8), acc1);
        } else {
            _mm256_storeu_ps(c_ptr.add(j), acc0);
            _mm256_storeu_ps(c_ptr.add(j + 8), acc1);
        }
        j += 16;
    }

    if j + 8 <= n {
        let mut acc = _mm256_setzero_ps();
        let c_block_aligned = (c_ptr.add(j) as usize).is_multiple_of(32);
        let mut kk = 0usize;
        // Peel until A is aligned
        while kk < k && !(a_ptr.add(kk) as usize).is_multiple_of(32) {
            let as_broadcast = _mm256_set1_ps(*a_ptr.add(kk));
            let b_row = b_ptr.add(kk * n + j);
            let vb = _mm256_loadu_ps(b_row);
            acc = _mm256_fmadd_ps(as_broadcast, vb, acc);
            kk += 1;
        }
        while kk + 8 <= k {
            let a8 = _mm256_load_ps(a_ptr.add(kk));
            let a_lane: [f32; 8] = core::mem::transmute(a8);
            for (t, &a_lane_t) in a_lane.iter().enumerate() {
                let b_row = b_ptr.add((kk + t) * n + j);
                let vb = if (b_row as usize).is_multiple_of(32) {
                    _mm256_load_ps(b_row)
                } else {
                    _mm256_loadu_ps(b_row)
                };
                let as_broadcast = _mm256_set1_ps(a_lane_t);
                acc = _mm256_fmadd_ps(as_broadcast, vb, acc);
            }
            kk += 8;
        }
        while kk < k {
            let as_broadcast = _mm256_set1_ps(*a_ptr.add(kk));
            let b_row = b_ptr.add(kk * n + j);
            let vb = if (b_row as usize).is_multiple_of(32) {
                _mm256_load_ps(b_row)
            } else {
                _mm256_loadu_ps(b_row)
            };
            acc = _mm256_fmadd_ps(as_broadcast, vb, acc);
            kk += 1;
        }
        if c_block_aligned {
            _mm256_store_ps(c_ptr.add(j), acc);
        } else {
            _mm256_storeu_ps(c_ptr.add(j), acc);
        }
        j += 8;
    }

    while j < n {
        let mut sum = 0.0f32;
        for kk in 0..k {
            sum += *a_ptr.add(kk) * *b_ptr.add(kk * n + j);
        }
        *c_ptr.add(j) = sum;
        j += 1;
    }
}

/// M (M,K) @ v (K) -> out (M)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_vec_avx2_unaligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
) {
    for i in 0..m {
        let row_ptr = a_ptr.add(i * k);
        let sum = dot_1d_avx2_unaligned(row_ptr, b_ptr, k);
        *c_ptr.add(i) = sum;
    }
}

/// AVX2 aligned M@v (delegates to unaligned for safety)
#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_vec_avx2_aligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
) {
    for i in 0..m {
        let row_ptr = a_ptr.add(i * k);
        // Per-row alignment check; fall back to unaligned if needed
        let sum = if (row_ptr as usize).is_multiple_of(32) {
            dot_1d_avx2_aligned(row_ptr, b_ptr, k)
        } else {
            dot_1d_avx2_unaligned(row_ptr, b_ptr, k)
        };
        *c_ptr.add(i) = sum;
    }
}

// ==========================
// Matrix-Matrix (2D @ 2D)
// ==========================

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn microkernel_6x16_direct(
    a_ptr: *const f32, // points to A at row i, col 0
    b_ptr: *const f32, // points to B at row 0, col j
    c_ptr: *mut f32,   // points to C at row i, col j
    k: usize,
    n: usize,
    valid_m: usize,
    valid_n: usize,
) {
    let mut r0_lo = _mm256_setzero_ps();
    let mut r1_lo = _mm256_setzero_ps();
    let mut r2_lo = _mm256_setzero_ps();
    let mut r3_lo = _mm256_setzero_ps();
    let mut r4_lo = _mm256_setzero_ps();
    let mut r5_lo = _mm256_setzero_ps();

    let mut r0_hi = _mm256_setzero_ps();
    let mut r1_hi = _mm256_setzero_ps();
    let mut r2_hi = _mm256_setzero_ps();
    let mut r3_hi = _mm256_setzero_ps();
    let mut r4_hi = _mm256_setzero_ps();
    let mut r5_hi = _mm256_setzero_ps();

    let mut kk = 0usize;
    while kk < k {
        // Prefetch next B row
        #[allow(unused_unsafe)]
        unsafe {
            if kk + 1 < k {
                _mm_prefetch(b_ptr.add((kk + 1) * n) as *const i8, _MM_HINT_T0);
            }
        }

        let b_row = b_ptr.add(kk * n);
        let b_lo = _mm256_loadu_ps(b_row);
        let b_hi = _mm256_loadu_ps(b_row.add(8));

        // Broadcast A scalars and FMA
        if valid_m > 0 {
            let a0 = _mm256_set1_ps(*a_ptr.add(kk));
            r0_lo = _mm256_fmadd_ps(a0, b_lo, r0_lo);
            r0_hi = _mm256_fmadd_ps(a0, b_hi, r0_hi);
        }
        if valid_m > 1 {
            let a1 = _mm256_set1_ps(*a_ptr.add(k + kk));
            r1_lo = _mm256_fmadd_ps(a1, b_lo, r1_lo);
            r1_hi = _mm256_fmadd_ps(a1, b_hi, r1_hi);
        }
        if valid_m > 2 {
            let a2 = _mm256_set1_ps(*a_ptr.add(2 * k + kk));
            r2_lo = _mm256_fmadd_ps(a2, b_lo, r2_lo);
            r2_hi = _mm256_fmadd_ps(a2, b_hi, r2_hi);
        }
        if valid_m > 3 {
            let a3 = _mm256_set1_ps(*a_ptr.add(3 * k + kk));
            r3_lo = _mm256_fmadd_ps(a3, b_lo, r3_lo);
            r3_hi = _mm256_fmadd_ps(a3, b_hi, r3_hi);
        }
        if valid_m > 4 {
            let a4 = _mm256_set1_ps(*a_ptr.add(4 * k + kk));
            r4_lo = _mm256_fmadd_ps(a4, b_lo, r4_lo);
            r4_hi = _mm256_fmadd_ps(a4, b_hi, r4_hi);
        }
        if valid_m > 5 {
            let a5 = _mm256_set1_ps(*a_ptr.add(5 * k + kk));
            r5_lo = _mm256_fmadd_ps(a5, b_lo, r5_lo);
            r5_hi = _mm256_fmadd_ps(a5, b_hi, r5_hi);
        }
        kk += 1;
    }

    // Store to C (assignment)
    let store_row = |vlo: __m256, vhi: __m256, c_row: *mut f32| {
        let mut lo: [f32; 8] = core::mem::zeroed();
        let mut hi: [f32; 8] = core::mem::zeroed();
        _mm256_storeu_ps(lo.as_mut_ptr(), vlo);
        _mm256_storeu_ps(hi.as_mut_ptr(), vhi);
        let left = valid_n.min(8);
        for (j, &lo_j) in lo.iter().enumerate().take(left) {
            *c_row.add(j) = lo_j;
        }
        if valid_n > 8 {
            let right = valid_n - 8;
            for (j, &hi_j) in hi.iter().enumerate().take(right) {
                *c_row.add(8 + j) = hi_j;
            }
        }
    };

    if valid_m > 0 {
        store_row(r0_lo, r0_hi, c_ptr);
    }
    if valid_m > 1 {
        store_row(r1_lo, r1_hi, c_ptr.add(n));
    }
    if valid_m > 2 {
        store_row(r2_lo, r2_hi, c_ptr.add(2 * n));
    }
    if valid_m > 3 {
        store_row(r3_lo, r3_hi, c_ptr.add(3 * n));
    }
    if valid_m > 4 {
        store_row(r4_lo, r4_hi, c_ptr.add(4 * n));
    }
    if valid_m > 5 {
        store_row(r5_lo, r5_hi, c_ptr.add(5 * n));
    }
}

// ==========================
// Packed GEMM micro-kernels
// ==========================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x16_packed_store(
    r_lo: [__m256; 6],
    r_hi: [__m256; 6],
    c_ptr: *mut f32,
    n_stride: usize,
    valid_m: usize,
    accumulate: bool,
) {
    for row in 0..valid_m {
        let crow = c_ptr.add(row * n_stride);
        if accumulate {
            let prev_lo = _mm256_loadu_ps(crow);
            let prev_hi = _mm256_loadu_ps(crow.add(8));
            let sum_lo = _mm256_add_ps(prev_lo, r_lo[row]);
            let sum_hi = _mm256_add_ps(prev_hi, r_hi[row]);
            _mm256_storeu_ps(crow, sum_lo);
            _mm256_storeu_ps(crow.add(8), sum_hi);
        } else {
            // For very large N, prefer streaming stores to reduce cache pollution
            let use_streaming = n_stride >= 1024; // heuristic threshold
            if use_streaming && (crow as usize).is_multiple_of(32) {
                _mm256_stream_ps(crow, r_lo[row]);
                _mm256_stream_ps(crow.add(8), r_hi[row]);
            } else {
                _mm256_storeu_ps(crow, r_lo[row]);
                _mm256_storeu_ps(crow.add(8), r_hi[row]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x16_packed_store_aligned(
    r_lo: [__m256; 6],
    r_hi: [__m256; 6],
    c_ptr: *mut f32,
    n_stride: usize,
    valid_m: usize,
    accumulate: bool,
) {
    for row in 0..valid_m {
        let crow = c_ptr.add(row * n_stride);
        if accumulate {
            let prev_lo = _mm256_load_ps(crow);
            let prev_hi = _mm256_load_ps(crow.add(8));
            let sum_lo = _mm256_add_ps(prev_lo, r_lo[row]);
            let sum_hi = _mm256_add_ps(prev_hi, r_hi[row]);
            _mm256_store_ps(crow, sum_lo);
            _mm256_store_ps(crow.add(8), sum_hi);
        } else {
            // For very large N, prefer streaming stores to reduce cache pollution
            let use_streaming = n_stride >= 1024; // heuristic threshold
            if use_streaming {
                _mm256_stream_ps(crow, r_lo[row]);
                _mm256_stream_ps(crow.add(8), r_hi[row]);
            } else {
                _mm256_store_ps(crow, r_lo[row]);
                _mm256_store_ps(crow.add(8), r_hi[row]);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn load256(b_ptr: *const f32, aligned: bool) -> __m256 {
    if aligned {
        _mm256_load_ps(b_ptr)
    } else {
        _mm256_loadu_ps(b_ptr)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x16_packed(
    a_pack: *const f32, // layout: [kc][mr] contiguous per k of size mr
    b_pack: *const f32, // layout: [kc][nr] contiguous per k of size 16
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    accumulate: bool,
    b_aligned: bool,
    c_aligned: bool,
) {
    let mut r_lo = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];
    let mut r_hi = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];
    let small_k = kc <= 32;
    let pf = prefetch_distance_kc(kc);

    let mut kk = 0usize;
    if small_k && kc >= 4 {
        // Unroll by 4 for small kc to reduce loop control overhead and hide load latency
        while kk + 3 < kc {
            #[allow(unused_unsafe)]
            unsafe {
                if pf > 0 && kk + pf < kc {
                    _mm_prefetch(b_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_pack.add((kk + pf) * 6) as *const i8, _MM_HINT_T0);
                }
            }

            let b0_lo = load256(b_pack.add(kk * 16), b_aligned);
            let b0_hi = load256(b_pack.add(kk * 16 + 8), b_aligned);
            let abase0 = a_pack.add(kk * 6);

            let b1_lo = load256(b_pack.add((kk + 1) * 16), b_aligned);
            let b1_hi = load256(b_pack.add((kk + 1) * 16 + 8), b_aligned);
            let abase1 = a_pack.add((kk + 1) * 6);

            let b2_lo = load256(b_pack.add((kk + 2) * 16), b_aligned);
            let b2_hi = load256(b_pack.add((kk + 2) * 16 + 8), b_aligned);
            let abase2 = a_pack.add((kk + 2) * 6);

            let b3_lo = load256(b_pack.add((kk + 3) * 16), b_aligned);
            let b3_hi = load256(b_pack.add((kk + 3) * 16 + 8), b_aligned);
            let abase3 = a_pack.add((kk + 3) * 6);

            let mut step = |abase: *const f32, blo: __m256, bhi: __m256| {
                if valid_m > 0 {
                    let a = _mm256_set1_ps(*abase.add(0));
                    r_lo[0] = _mm256_fmadd_ps(a, blo, r_lo[0]);
                    r_hi[0] = _mm256_fmadd_ps(a, bhi, r_hi[0]);
                }
                if valid_m > 1 {
                    let a = _mm256_set1_ps(*abase.add(1));
                    r_lo[1] = _mm256_fmadd_ps(a, blo, r_lo[1]);
                    r_hi[1] = _mm256_fmadd_ps(a, bhi, r_hi[1]);
                }
                if valid_m > 2 {
                    let a = _mm256_set1_ps(*abase.add(2));
                    r_lo[2] = _mm256_fmadd_ps(a, blo, r_lo[2]);
                    r_hi[2] = _mm256_fmadd_ps(a, bhi, r_hi[2]);
                }
                if valid_m > 3 {
                    let a = _mm256_set1_ps(*abase.add(3));
                    r_lo[3] = _mm256_fmadd_ps(a, blo, r_lo[3]);
                    r_hi[3] = _mm256_fmadd_ps(a, bhi, r_hi[3]);
                }
                if valid_m > 4 {
                    let a = _mm256_set1_ps(*abase.add(4));
                    r_lo[4] = _mm256_fmadd_ps(a, blo, r_lo[4]);
                    r_hi[4] = _mm256_fmadd_ps(a, bhi, r_hi[4]);
                }
                if valid_m > 5 {
                    let a = _mm256_set1_ps(*abase.add(5));
                    r_lo[5] = _mm256_fmadd_ps(a, blo, r_lo[5]);
                    r_hi[5] = _mm256_fmadd_ps(a, bhi, r_hi[5]);
                }
            };

            step(abase0, b0_lo, b0_hi);
            step(abase1, b1_lo, b1_hi);
            step(abase2, b2_lo, b2_hi);
            step(abase3, b3_lo, b3_hi);

            kk += 4;
        }
    } else if !small_k {
        while kk + 1 < kc {
            #[allow(unused_unsafe)]
            unsafe {
                if pf > 0 && kk + pf < kc {
                    _mm_prefetch(b_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_pack.add((kk + pf) * 6) as *const i8, _MM_HINT_T0);
                }
            }

            let b0_lo = load256(b_pack.add(kk * 16), b_aligned);
            let b0_hi = load256(b_pack.add(kk * 16 + 8), b_aligned);
            let b1_lo = load256(b_pack.add((kk + 1) * 16), b_aligned);
            let b1_hi = load256(b_pack.add((kk + 1) * 16 + 8), b_aligned);

            let abase0 = a_pack.add(kk * 6);
            let abase1 = a_pack.add((kk + 1) * 6);

            if valid_m > 0 {
                let a0 = _mm256_set1_ps(*abase0.add(0));
                let a1 = _mm256_set1_ps(*abase1.add(0));
                r_lo[0] = _mm256_fmadd_ps(a0, b0_lo, r_lo[0]);
                r_hi[0] = _mm256_fmadd_ps(a0, b0_hi, r_hi[0]);
                r_lo[0] = _mm256_fmadd_ps(a1, b1_lo, r_lo[0]);
                r_hi[0] = _mm256_fmadd_ps(a1, b1_hi, r_hi[0]);
            }
            if valid_m > 1 {
                let a0 = _mm256_set1_ps(*abase0.add(1));
                let a1 = _mm256_set1_ps(*abase1.add(1));
                r_lo[1] = _mm256_fmadd_ps(a0, b0_lo, r_lo[1]);
                r_hi[1] = _mm256_fmadd_ps(a0, b0_hi, r_hi[1]);
                r_lo[1] = _mm256_fmadd_ps(a1, b1_lo, r_lo[1]);
                r_hi[1] = _mm256_fmadd_ps(a1, b1_hi, r_hi[1]);
            }
            if valid_m > 2 {
                let a0 = _mm256_set1_ps(*abase0.add(2));
                let a1 = _mm256_set1_ps(*abase1.add(2));
                r_lo[2] = _mm256_fmadd_ps(a0, b0_lo, r_lo[2]);
                r_hi[2] = _mm256_fmadd_ps(a0, b0_hi, r_hi[2]);
                r_lo[2] = _mm256_fmadd_ps(a1, b1_lo, r_lo[2]);
                r_hi[2] = _mm256_fmadd_ps(a1, b1_hi, r_hi[2]);
            }
            if valid_m > 3 {
                let a0 = _mm256_set1_ps(*abase0.add(3));
                let a1 = _mm256_set1_ps(*abase1.add(3));
                r_lo[3] = _mm256_fmadd_ps(a0, b0_lo, r_lo[3]);
                r_hi[3] = _mm256_fmadd_ps(a0, b0_hi, r_hi[3]);
                r_lo[3] = _mm256_fmadd_ps(a1, b1_lo, r_lo[3]);
                r_hi[3] = _mm256_fmadd_ps(a1, b1_hi, r_hi[3]);
            }
            if valid_m > 4 {
                let a0 = _mm256_set1_ps(*abase0.add(4));
                let a1 = _mm256_set1_ps(*abase1.add(4));
                r_lo[4] = _mm256_fmadd_ps(a0, b0_lo, r_lo[4]);
                r_hi[4] = _mm256_fmadd_ps(a0, b0_hi, r_hi[4]);
                r_lo[4] = _mm256_fmadd_ps(a1, b1_lo, r_lo[4]);
                r_hi[4] = _mm256_fmadd_ps(a1, b1_hi, r_hi[4]);
            }
            if valid_m > 5 {
                let a0 = _mm256_set1_ps(*abase0.add(5));
                let a1 = _mm256_set1_ps(*abase1.add(5));
                r_lo[5] = _mm256_fmadd_ps(a0, b0_lo, r_lo[5]);
                r_hi[5] = _mm256_fmadd_ps(a0, b0_hi, r_hi[5]);
                r_lo[5] = _mm256_fmadd_ps(a1, b1_lo, r_lo[5]);
                r_hi[5] = _mm256_fmadd_ps(a1, b1_hi, r_hi[5]);
            }

            kk += 2;
        }
    }

    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + 1 < kc {
                _mm_prefetch(b_pack.add((kk + 1) * 16) as *const i8, _MM_HINT_T0);
            }
        }
        let b_lo = load256(b_pack.add(kk * 16), b_aligned);
        let b_hi = load256(b_pack.add(kk * 16 + 8), b_aligned);
        let abase = a_pack.add(kk * 6);
        if valid_m > 0 {
            let a = _mm256_set1_ps(*abase.add(0));
            r_lo[0] = _mm256_fmadd_ps(a, b_lo, r_lo[0]);
            r_hi[0] = _mm256_fmadd_ps(a, b_hi, r_hi[0]);
        }
        if valid_m > 1 {
            let a = _mm256_set1_ps(*abase.add(1));
            r_lo[1] = _mm256_fmadd_ps(a, b_lo, r_lo[1]);
            r_hi[1] = _mm256_fmadd_ps(a, b_hi, r_hi[1]);
        }
        if valid_m > 2 {
            let a = _mm256_set1_ps(*abase.add(2));
            r_lo[2] = _mm256_fmadd_ps(a, b_lo, r_lo[2]);
            r_hi[2] = _mm256_fmadd_ps(a, b_hi, r_hi[2]);
        }
        if valid_m > 3 {
            let a = _mm256_set1_ps(*abase.add(3));
            r_lo[3] = _mm256_fmadd_ps(a, b_lo, r_lo[3]);
            r_hi[3] = _mm256_fmadd_ps(a, b_hi, r_hi[3]);
        }
        if valid_m > 4 {
            let a = _mm256_set1_ps(*abase.add(4));
            r_lo[4] = _mm256_fmadd_ps(a, b_lo, r_lo[4]);
            r_hi[4] = _mm256_fmadd_ps(a, b_hi, r_hi[4]);
        }
        if valid_m > 5 {
            let a = _mm256_set1_ps(*abase.add(5));
            r_lo[5] = _mm256_fmadd_ps(a, b_lo, r_lo[5]);
            r_hi[5] = _mm256_fmadd_ps(a, b_hi, r_hi[5]);
        }
        kk += 1;
    }

    if c_aligned {
        microkernel_6x16_packed_store_aligned(r_lo, r_hi, c_ptr, n_stride, valid_m, accumulate);
    } else {
        microkernel_6x16_packed_store(r_lo, r_hi, c_ptr, n_stride, valid_m, accumulate);
    }
}

// (Removed) 8x16 packed microkernel variants were unused and not aligned with the
// chosen mr=6 blocking strategy. They have been removed to reduce binary size and
// avoid maintenance overhead. If future tuning indicates benefits for mr=8 paths,
// reintroduce with targeted dispatch based on M shape characteristics.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x8_packed(
    a_pack: *const f32,
    b_pack: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    valid_n: usize,
    accumulate: bool,
) {
    let mut r = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];
    let b_aligned = (b_pack as usize).is_multiple_of(32);
    let c_aligned = (c_ptr as usize).is_multiple_of(32) && n_stride.is_multiple_of(8);
    let pf = prefetch_distance_kc(kc);

    let mut kk = 0usize;
    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + pf < kc {
                _mm_prefetch(b_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                _mm_prefetch(a_pack.add((kk + pf) * 6) as *const i8, _MM_HINT_T0);
            }
        }
        let b = load256(b_pack.add(kk * 16), b_aligned); // first 8
        let abase = a_pack.add(kk * 6);
        if valid_m > 0 {
            let a = _mm256_set1_ps(*abase.add(0));
            r[0] = _mm256_fmadd_ps(a, b, r[0]);
        }
        if valid_m > 1 {
            let a = _mm256_set1_ps(*abase.add(1));
            r[1] = _mm256_fmadd_ps(a, b, r[1]);
        }
        if valid_m > 2 {
            let a = _mm256_set1_ps(*abase.add(2));
            r[2] = _mm256_fmadd_ps(a, b, r[2]);
        }
        if valid_m > 3 {
            let a = _mm256_set1_ps(*abase.add(3));
            r[3] = _mm256_fmadd_ps(a, b, r[3]);
        }
        if valid_m > 4 {
            let a = _mm256_set1_ps(*abase.add(4));
            r[4] = _mm256_fmadd_ps(a, b, r[4]);
        }
        if valid_m > 5 {
            let a = _mm256_set1_ps(*abase.add(5));
            r[5] = _mm256_fmadd_ps(a, b, r[5]);
        }
        kk += 1;
    }

    for (row, r_row) in r.iter().enumerate().take(valid_m) {
        let crow = c_ptr.add(row * n_stride);
        if accumulate {
            if c_aligned {
                let prev = _mm256_load_ps(crow);
                let sum = _mm256_add_ps(prev, *r_row);
                _mm256_store_ps(crow, sum);
            } else {
                let prev = _mm256_loadu_ps(crow);
                let sum = _mm256_add_ps(prev, *r_row);
                _mm256_storeu_ps(crow, sum);
            }
        } else if c_aligned {
            _mm256_store_ps(crow, *r_row);
        } else {
            _mm256_storeu_ps(crow, *r_row);
        }
    }

    if valid_n < 8 {
        let mut tmp = [0.0f32; 8];
        for (row, r_row) in r.iter().enumerate().take(valid_m) {
            _mm256_storeu_ps(tmp.as_mut_ptr(), *r_row);
            for (col, &tmp_col) in tmp.iter().enumerate().take(valid_n) {
                let dst = c_ptr.add(row * n_stride + col);
                if accumulate {
                    *dst += tmp_col;
                } else {
                    *dst = tmp_col;
                }
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x8_packed_aligned(
    a_pack: *const f32,
    b_pack: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    valid_n: usize,
    accumulate: bool,
) {
    let mut r = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];

    let b_aligned = (b_pack as usize).is_multiple_of(32);
    let pf = prefetch_distance_kc(kc);

    let mut kk = 0usize;
    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + pf < kc {
                _mm_prefetch(b_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                _mm_prefetch(a_pack.add((kk + pf) * 6) as *const i8, _MM_HINT_T0);
            }
        }
        // B-panel now can use aligned loads when packing alignment allows
        let b = load256(b_pack.add(kk * 16), b_aligned);
        let abase = a_pack.add(kk * 6);
        if valid_m > 0 {
            let a = _mm256_set1_ps(*abase.add(0));
            r[0] = _mm256_fmadd_ps(a, b, r[0]);
        }
        if valid_m > 1 {
            let a = _mm256_set1_ps(*abase.add(1));
            r[1] = _mm256_fmadd_ps(a, b, r[1]);
        }
        if valid_m > 2 {
            let a = _mm256_set1_ps(*abase.add(2));
            r[2] = _mm256_fmadd_ps(a, b, r[2]);
        }
        if valid_m > 3 {
            let a = _mm256_set1_ps(*abase.add(3));
            r[3] = _mm256_fmadd_ps(a, b, r[3]);
        }
        if valid_m > 4 {
            let a = _mm256_set1_ps(*abase.add(4));
            r[4] = _mm256_fmadd_ps(a, b, r[4]);
        }
        if valid_m > 5 {
            let a = _mm256_set1_ps(*abase.add(5));
            r[5] = _mm256_fmadd_ps(a, b, r[5]);
        }
        kk += 1;
    }

    for (row, r_row) in r.iter().enumerate().take(valid_m) {
        let crow = c_ptr.add(row * n_stride);
        if accumulate {
            let prev = _mm256_load_ps(crow);
            let sum = _mm256_add_ps(prev, *r_row);
            _mm256_store_ps(crow, sum);
        } else {
            _mm256_store_ps(crow, *r_row);
        }
    }

    if valid_n < 8 {
        let mut tmp = [0.0f32; 8];
        for (row, r_row) in r.iter().enumerate().take(valid_m) {
            _mm256_storeu_ps(tmp.as_mut_ptr(), *r_row);
            for (col, &tmp_col) in tmp.iter().enumerate().take(valid_n) {
                let dst = c_ptr.add(row * n_stride + col);
                if accumulate {
                    *dst += tmp_col;
                } else {
                    *dst = tmp_col;
                }
            }
        }
    }
}

// 6x12 packed microkernel: computes 12 columns (8 + 4) without external splitting
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x12_packed(
    a_pack: *const f32, // layout: [kc][mr] contiguous per k of size 6
    b_pack: *const f32, // layout: [kc][nr] contiguous per k of size 16 (last 4 are padded zero)
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize, // may be 12 (contiguous) or 16 (tile)
    valid_m: usize,
    accumulate: bool,
    b_aligned: bool,
    c_aligned32: bool,
) {
    let mut r_lo = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];
    let mut r_hi = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ];

    let pf = prefetch_distance_kc(kc);
    let mut kk = 0usize;
    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + 1 < kc {
                _mm_prefetch(b_pack.add((kk + 1) * 16) as *const i8, _MM_HINT_T0);
                _mm_prefetch(a_pack.add((kk + 1) * 6) as *const i8, _MM_HINT_T0);
            }
        }
        // Load 16 values from B panel (last 4 may be padded zeros); accumulate into 8+8
        let b_lo = load256(b_pack.add(kk * 16), b_aligned);
        let b_hi = load256(b_pack.add(kk * 16 + 8), b_aligned);
        let abase = a_pack.add(kk * 6);
        if valid_m > 0 {
            let a = _mm256_set1_ps(*abase.add(0));
            r_lo[0] = _mm256_fmadd_ps(a, b_lo, r_lo[0]);
            r_hi[0] = _mm256_fmadd_ps(a, b_hi, r_hi[0]);
        }
        if valid_m > 1 {
            let a = _mm256_set1_ps(*abase.add(1));
            r_lo[1] = _mm256_fmadd_ps(a, b_lo, r_lo[1]);
            r_hi[1] = _mm256_fmadd_ps(a, b_hi, r_hi[1]);
        }
        if valid_m > 2 {
            let a = _mm256_set1_ps(*abase.add(2));
            r_lo[2] = _mm256_fmadd_ps(a, b_lo, r_lo[2]);
            r_hi[2] = _mm256_fmadd_ps(a, b_hi, r_hi[2]);
        }
        if valid_m > 3 {
            let a = _mm256_set1_ps(*abase.add(3));
            r_lo[3] = _mm256_fmadd_ps(a, b_lo, r_lo[3]);
            r_hi[3] = _mm256_fmadd_ps(a, b_hi, r_hi[3]);
        }
        if valid_m > 4 {
            let a = _mm256_set1_ps(*abase.add(4));
            r_lo[4] = _mm256_fmadd_ps(a, b_lo, r_lo[4]);
            r_hi[4] = _mm256_fmadd_ps(a, b_hi, r_hi[4]);
        }
        if valid_m > 5 {
            let a = _mm256_set1_ps(*abase.add(5));
            r_lo[5] = _mm256_fmadd_ps(a, b_lo, r_lo[5]);
            r_hi[5] = _mm256_fmadd_ps(a, b_hi, r_hi[5]);
        }
        kk += 1;
    }

    // Store 12 columns = 8 from r_lo and 4 from low 128 of r_hi
    let use_streaming = n_stride >= 1024;
    for row in 0..valid_m {
        let crow = c_ptr.add(row * n_stride);
        // 8-wide store
        if accumulate {
            if c_aligned32 {
                let prev = _mm256_load_ps(crow);
                let sum = _mm256_add_ps(prev, r_lo[row]);
                _mm256_store_ps(crow, sum);
            } else {
                let prev = _mm256_loadu_ps(crow);
                let sum = _mm256_add_ps(prev, r_lo[row]);
                _mm256_storeu_ps(crow, sum);
            }
        } else if use_streaming && c_aligned32 {
            _mm256_stream_ps(crow, r_lo[row]);
        } else if c_aligned32 {
            _mm256_store_ps(crow, r_lo[row]);
        } else {
            _mm256_storeu_ps(crow, r_lo[row]);
        }

        // 4-wide tail at crow+8
        let crow_tail = crow.add(8);
        // Extract low 128-bit lane (cols 8..11)
        let tail128: __m128 = _mm256_castps256_ps128(r_hi[row]);
        if accumulate {
            let prev = _mm_loadu_ps(crow_tail);
            let sum = _mm_add_ps(prev, tail128);
            _mm_storeu_ps(crow_tail, sum);
        } else {
            // Try aligned 16 store if possible, else unaligned
            if (crow_tail as usize).is_multiple_of(16) && use_streaming {
                // No streaming intrinsic for 128 in stable; fall back to aligned store
                _mm_store_ps(crow_tail, tail128);
            } else if (crow_tail as usize).is_multiple_of(16) {
                _mm_store_ps(crow_tail, tail128);
            } else {
                _mm_storeu_ps(crow_tail, tail128);
            }
        }
    }
}

// 6x4 packed microkernel to reduce scalar tail work for N tails of 4 columns
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x4_packed(
    a_pack: *const f32, // layout: [kc][mr] contiguous per k of size 6
    b_pack: *const f32, // layout: [kc][nr] contiguous per k of size 16 (first 4 valid)
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    accumulate: bool,
    b_aligned: bool,
    c_aligned: bool, // 16-byte alignment for 4-wide stores
) {
    let mut r = [
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
        _mm_setzero_ps(),
    ];

    let pf = prefetch_distance_kc(kc);

    // Helper to load 4 floats from packed B (first 4 of 16)
    #[inline(always)]
    unsafe fn load128(ptr: *const f32, aligned: bool) -> __m128 {
        if aligned {
            _mm_load_ps(ptr)
        } else {
            _mm_loadu_ps(ptr)
        }
    }

    let mut kk = 0usize;
    // Unroll by 4 when possible to reduce loop overhead and handle K-tail in-line
    while kk + 3 < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + pf < kc {
                _mm_prefetch(b_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
            }
        }

        let b0 = load128(b_pack.add(kk * 16), b_aligned);
        let abase0 = a_pack.add(kk * 6);

        let b1 = load128(b_pack.add((kk + 1) * 16), b_aligned);
        let abase1 = a_pack.add((kk + 1) * 6);

        let b2 = load128(b_pack.add((kk + 2) * 16), b_aligned);
        let abase2 = a_pack.add((kk + 2) * 6);

        let b3 = load128(b_pack.add((kk + 3) * 16), b_aligned);
        let abase3 = a_pack.add((kk + 3) * 6);

        let mut step = |abase: *const f32, bvec: __m128| {
            for (row, r_row) in r.iter_mut().enumerate().take(valid_m) {
                let a = _mm_set1_ps(*abase.add(row));
                *r_row = _mm_fmadd_ps(a, bvec, *r_row);
            }
        };

        step(abase0, b0);
        step(abase1, b1);
        step(abase2, b2);
        step(abase3, b3);

        kk += 4;
    }

    // Handle remaining K tail (0..=3) in-line
    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + 1 < kc {
                _mm_prefetch(b_pack.add((kk + 1) * 16) as *const i8, _MM_HINT_T0);
            }
        }
        let b = load128(b_pack.add(kk * 16), b_aligned);
        let abase = a_pack.add(kk * 6);
        for (row, r_row) in r.iter_mut().enumerate().take(valid_m) {
            let a = _mm_set1_ps(*abase.add(row));
            *r_row = _mm_fmadd_ps(a, b, *r_row);
        }
        kk += 1;
    }

    // Store 4 columns to C
    for (row, r_row) in r.iter().enumerate().take(valid_m) {
        let crow = c_ptr.add(row * n_stride);
        if accumulate {
            let prev = if c_aligned {
                _mm_load_ps(crow)
            } else {
                _mm_loadu_ps(crow)
            };
            let sum = _mm_add_ps(prev, *r_row);
            if c_aligned {
                _mm_store_ps(crow, sum);
            } else {
                _mm_storeu_ps(crow, sum);
            }
        } else if c_aligned {
            _mm_store_ps(crow, *r_row);
        } else {
            _mm_storeu_ps(crow, *r_row);
        }
    }
}

// (Removed) 12x16 packed microkernel was unused. Current packing uses mr=6 paths
// consistently; combining two 6-row blocks is handled by outer loops.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x_tail_scalar(
    a_pack: *const f32,
    b_pack: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    n_tail: usize,
    b_col_offset: usize,
    accumulate: bool,
) {
    for row in 0..valid_m {
        for col in 0..n_tail {
            let mut sum = 0.0f32;
            for kk in 0..kc {
                let a = *a_pack.add(kk * 6 + row);
                let b = *b_pack.add(kk * 16 + b_col_offset + col);
                sum += a * b;
            }
            let dst = c_ptr.add(row * n_stride + col);
            if accumulate {
                *dst += sum;
            } else {
                *dst = sum;
            }
        }
    }
}

// 6x24 fused microkernel using two B panels (16 + 8) to reuse A broadcasts
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn microkernel_6x24_packed(
    a_pack: *const f32,
    b16_pack: *const f32,
    b8_pack: *const f32,
    c_ptr: *mut f32,
    kc: usize,
    n_stride: usize,
    valid_m: usize,
    accumulate: bool,
    b16_aligned: bool,
    b8_aligned: bool,
    c_aligned: bool,
) {
    let mut r0 = [
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
        _mm256_setzero_ps(),
    ]; // cols 0..7
    let mut r1 = r0; // cols 8..15
    let mut r2 = r0; // cols 16..23
    let small_k = kc <= 64;
    let pf = prefetch_distance_kc(kc);

    let mut kk = 0usize;
    if small_k && kc >= 4 {
        while kk + 3 < kc {
            #[allow(unused_unsafe)]
            unsafe {
                if pf > 0 && kk + pf < kc {
                    _mm_prefetch(b16_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(b8_pack.add((kk + pf) * 16) as *const i8, _MM_HINT_T0);
                    _mm_prefetch(a_pack.add((kk + pf) * 6) as *const i8, _MM_HINT_T0);
                }
            }

            let mut step = |abase: *const f32, b16_lo: __m256, b16_hi: __m256, b8: __m256| {
                if valid_m > 0 {
                    let a = _mm256_set1_ps(*abase.add(0));
                    r0[0] = _mm256_fmadd_ps(a, b16_lo, r0[0]);
                    r1[0] = _mm256_fmadd_ps(a, b16_hi, r1[0]);
                    r2[0] = _mm256_fmadd_ps(a, b8, r2[0]);
                }
                if valid_m > 1 {
                    let a = _mm256_set1_ps(*abase.add(1));
                    r0[1] = _mm256_fmadd_ps(a, b16_lo, r0[1]);
                    r1[1] = _mm256_fmadd_ps(a, b16_hi, r1[1]);
                    r2[1] = _mm256_fmadd_ps(a, b8, r2[1]);
                }
                if valid_m > 2 {
                    let a = _mm256_set1_ps(*abase.add(2));
                    r0[2] = _mm256_fmadd_ps(a, b16_lo, r0[2]);
                    r1[2] = _mm256_fmadd_ps(a, b16_hi, r1[2]);
                    r2[2] = _mm256_fmadd_ps(a, b8, r2[2]);
                }
                if valid_m > 3 {
                    let a = _mm256_set1_ps(*abase.add(3));
                    r0[3] = _mm256_fmadd_ps(a, b16_lo, r0[3]);
                    r1[3] = _mm256_fmadd_ps(a, b16_hi, r1[3]);
                    r2[3] = _mm256_fmadd_ps(a, b8, r2[3]);
                }
                if valid_m > 4 {
                    let a = _mm256_set1_ps(*abase.add(4));
                    r0[4] = _mm256_fmadd_ps(a, b16_lo, r0[4]);
                    r1[4] = _mm256_fmadd_ps(a, b16_hi, r1[4]);
                    r2[4] = _mm256_fmadd_ps(a, b8, r2[4]);
                }
                if valid_m > 5 {
                    let a = _mm256_set1_ps(*abase.add(5));
                    r0[5] = _mm256_fmadd_ps(a, b16_lo, r0[5]);
                    r1[5] = _mm256_fmadd_ps(a, b16_hi, r1[5]);
                    r2[5] = _mm256_fmadd_ps(a, b8, r2[5]);
                }
            };

            // four steps
            let b0_lo = load256(b16_pack.add(kk * 16), b16_aligned);
            let b0_hi = load256(b16_pack.add(kk * 16 + 8), b16_aligned);
            let b0_8 = load256(b8_pack.add(kk * 16), b8_aligned); // first 8 cols packed into first lane
            step(a_pack.add(kk * 6), b0_lo, b0_hi, b0_8);

            let b1_lo = load256(b16_pack.add((kk + 1) * 16), b16_aligned);
            let b1_hi = load256(b16_pack.add((kk + 1) * 16 + 8), b16_aligned);
            let b1_8 = load256(b8_pack.add((kk + 1) * 16), b8_aligned);
            step(a_pack.add((kk + 1) * 6), b1_lo, b1_hi, b1_8);

            let b2_lo = load256(b16_pack.add((kk + 2) * 16), b16_aligned);
            let b2_hi = load256(b16_pack.add((kk + 2) * 16 + 8), b16_aligned);
            let b2_8 = load256(b8_pack.add((kk + 2) * 16), b8_aligned);
            step(a_pack.add((kk + 2) * 6), b2_lo, b2_hi, b2_8);

            let b3_lo = load256(b16_pack.add((kk + 3) * 16), b16_aligned);
            let b3_hi = load256(b16_pack.add((kk + 3) * 16 + 8), b16_aligned);
            let b3_8 = load256(b8_pack.add((kk + 3) * 16), b8_aligned);
            step(a_pack.add((kk + 3) * 6), b3_lo, b3_hi, b3_8);

            kk += 4;
        }
    }

    while kk < kc {
        #[allow(unused_unsafe)]
        unsafe {
            if pf > 0 && kk + 1 < kc {
                _mm_prefetch(b16_pack.add((kk + 1) * 16) as *const i8, _MM_HINT_T0);
                _mm_prefetch(b8_pack.add((kk + 1) * 16) as *const i8, _MM_HINT_T0);
            }
        }
        let b16_lo = load256(b16_pack.add(kk * 16), b16_aligned);
        let b16_hi = load256(b16_pack.add(kk * 16 + 8), b16_aligned);
        let b8 = load256(b8_pack.add(kk * 16), b8_aligned);
        let abase = a_pack.add(kk * 6);
        if valid_m > 0 {
            let a = _mm256_set1_ps(*abase.add(0));
            r0[0] = _mm256_fmadd_ps(a, b16_lo, r0[0]);
            r1[0] = _mm256_fmadd_ps(a, b16_hi, r1[0]);
            r2[0] = _mm256_fmadd_ps(a, b8, r2[0]);
        }
        if valid_m > 1 {
            let a = _mm256_set1_ps(*abase.add(1));
            r0[1] = _mm256_fmadd_ps(a, b16_lo, r0[1]);
            r1[1] = _mm256_fmadd_ps(a, b16_hi, r1[1]);
            r2[1] = _mm256_fmadd_ps(a, b8, r2[1]);
        }
        if valid_m > 2 {
            let a = _mm256_set1_ps(*abase.add(2));
            r0[2] = _mm256_fmadd_ps(a, b16_lo, r0[2]);
            r1[2] = _mm256_fmadd_ps(a, b16_hi, r1[2]);
            r2[2] = _mm256_fmadd_ps(a, b8, r2[2]);
        }
        if valid_m > 3 {
            let a = _mm256_set1_ps(*abase.add(3));
            r0[3] = _mm256_fmadd_ps(a, b16_lo, r0[3]);
            r1[3] = _mm256_fmadd_ps(a, b16_hi, r1[3]);
            r2[3] = _mm256_fmadd_ps(a, b8, r2[3]);
        }
        if valid_m > 4 {
            let a = _mm256_set1_ps(*abase.add(4));
            r0[4] = _mm256_fmadd_ps(a, b16_lo, r0[4]);
            r1[4] = _mm256_fmadd_ps(a, b16_hi, r1[4]);
            r2[4] = _mm256_fmadd_ps(a, b8, r2[4]);
        }
        if valid_m > 5 {
            let a = _mm256_set1_ps(*abase.add(5));
            r0[5] = _mm256_fmadd_ps(a, b16_lo, r0[5]);
            r1[5] = _mm256_fmadd_ps(a, b16_hi, r1[5]);
            r2[5] = _mm256_fmadd_ps(a, b8, r2[5]);
        }
        kk += 1;
    }

    // Store 24 columns: first 16 then 8
    if c_aligned {
        microkernel_6x16_packed_store_aligned(r0, r1, c_ptr, n_stride, valid_m, accumulate);
    } else {
        microkernel_6x16_packed_store(r0, r1, c_ptr, n_stride, valid_m, accumulate);
    }
    // Next 8 columns
    for (row, r_row) in r2.iter().enumerate().take(valid_m) {
        let crow = c_ptr.add(row * n_stride + 16);
        if accumulate {
            let prev = _mm256_loadu_ps(crow);
            let sum = _mm256_add_ps(prev, *r_row);
            _mm256_storeu_ps(crow, sum);
        } else {
            _mm256_storeu_ps(crow, *r_row);
        }
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_small_avx2_unaligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    // 2D tiling across N for better locality when writing C
    // Use a smaller nc for very small problems to reduce loop/setup overhead
    let total = m * k + k * n + m * n;
    let nc_block = if total <= 4096 { 128usize } else { 256usize };
    let mut jc = 0usize;
    while jc < n {
        let nc_curr = (n - jc).min(nc_block);
        let mut i = 0usize;
        while i < m {
            let valid_m = (m - i).min(6);
            let mut j = 0usize;
            while j < nc_curr {
                let valid_n = (nc_curr - j).min(16);
                microkernel_6x16_direct(
                    a_ptr.add(i * k),
                    b_ptr.add(jc + j),
                    c_ptr.add(i * n + jc + j),
                    k,
                    n,
                    valid_m,
                    valid_n,
                );
                j += 16;
            }
            i += 6;
        }
        jc += nc_block;
    }
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_medium_avx2_unaligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    packed_gemm_avx2(a_ptr, b_ptr, c_ptr, m, k, n)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_large_avx2_unaligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    packed_gemm_avx2(a_ptr, b_ptr, c_ptr, m, k, n)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_small_avx2_aligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    mat_mat_small_avx2_unaligned(a_ptr, b_ptr, c_ptr, m, k, n)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_medium_avx2_aligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    packed_gemm_avx2(a_ptr, b_ptr, c_ptr, m, k, n)
}

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_large_avx2_aligned(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    packed_gemm_avx2(a_ptr, b_ptr, c_ptr, m, k, n)
}

// ==========================
// Packed GEMM (panels + blocking)
// ==========================

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packed_gemm_avx2(
    a_ptr: *const f32,
    b_ptr: *const f32,
    c_ptr: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    let params = BlockingParams::get_cached();
    let kc_block = params.kc;
    let mc_block = params.mc;
    let mr = 6usize; // AVX2
    let nc_block = 256usize; // N-tiling for better locality

    let mut pc = 0usize;
    while pc < k {
        let kc_curr = (k - pc).min(kc_block);

        let mut jc = 0usize;
        while jc < n {
            let nc_curr = (n - jc).min(nc_block);

            let mut jr = 0usize;
            while jr < nc_curr {
                let remain_n = nc_curr - jr;
                // Try fused 24-wide path when beneficial
                if remain_n >= 24 {
                    with_panel_cache(|cache| {
                        let (b16_pack_ptr, b8_pack_ptr) = unsafe {
                            // Pack 16 columns
                            let p16 = {
                                let panel_b = cache.get_or_create_panel_b(kc_curr, 16);
                                panel_b.pack_from_matrix(
                                    b_ptr.add(pc * n + jc + jr),
                                    n,
                                    kc_curr,
                                    16,
                                );
                                panel_b.as_ptr()
                            };
                            // Pack next 8 columns
                            let p8 = {
                                let panel_b = cache.get_or_create_panel_b(kc_curr, 8);
                                panel_b.pack_from_matrix(
                                    b_ptr.add(pc * n + jc + jr + 16),
                                    n,
                                    kc_curr,
                                    8,
                                );
                                panel_b.as_ptr()
                            };
                            (p16, p8)
                        };
                        let b16_aligned = (b16_pack_ptr as usize).is_multiple_of(32);
                        let b8_aligned = (b8_pack_ptr as usize).is_multiple_of(32);

                        let mut ic = 0usize;
                        while ic < m {
                            let block_m = (m - ic).min(mc_block);

                            // Double-buffer A micro-panels: pack one while computing the other
                            let mut a_buf0 = PackedPanelA::new(mr, kc_curr);
                            let mut a_buf1 = PackedPanelA::new(mr, kc_curr);
                            let mut use_buf0 = true;

                            // Iterate rows in mr blocks
                            let mut i_block = 0usize;
                            // Pre-pack first micro-block if any
                            if block_m > 0 {
                                let valid_m0 = block_m.min(mr);
                                let a_src0 = a_ptr.add(ic * k + pc);
                                a_buf0.pack_from_matrix(a_src0, k, valid_m0, kc_curr);
                            }

                            while i_block < block_m {
                                let remain_m = block_m - i_block;
                                let valid_m = remain_m.min(mr);
                                let c_micro = c_ptr.add((ic + i_block) * n + jc + jr);
                                let accumulate = pc != 0;
                                let c_aligned = (c_micro as usize).is_multiple_of(32)
                                    && n.is_multiple_of(8)
                                    && jr.is_multiple_of(8);

                                // Pre-pack next micro-block now
                                let has_next = i_block + mr < block_m;
                                if has_next {
                                    let next_valid_m = (block_m - (i_block + mr)).min(mr);
                                    let a_src_next = a_ptr.add((ic + i_block + mr) * k + pc);
                                    if use_buf0 {
                                        a_buf1.pack_from_matrix(
                                            a_src_next,
                                            k,
                                            next_valid_m,
                                            kc_curr,
                                        );
                                    } else {
                                        a_buf0.pack_from_matrix(
                                            a_src_next,
                                            k,
                                            next_valid_m,
                                            kc_curr,
                                        );
                                    }
                                }

                                let a_micro = if use_buf0 {
                                    a_buf0.as_ptr()
                                } else {
                                    a_buf1.as_ptr()
                                };

                                microkernel_6x24_packed(
                                    a_micro,
                                    b16_pack_ptr,
                                    b8_pack_ptr,
                                    c_micro,
                                    kc_curr,
                                    n,
                                    valid_m,
                                    accumulate,
                                    b16_aligned,
                                    b8_aligned,
                                    c_aligned,
                                );

                                use_buf0 = !use_buf0;
                                i_block += mr;
                            }

                            ic += mc_block;
                        }
                    });
                    jr += 24;
                    continue;
                }

                // Segment remaining N into {16, 8, 4} chunks to minimize tails
                with_panel_cache(|cache| {
                    let mut seg_offset = 0usize;
                    while seg_offset < remain_n {
                        let seg_left = remain_n - seg_offset;
                        let seg_n = if seg_left >= 16 {
                            16
                        } else if seg_left >= 12 {
                            12
                        } else if seg_left >= 8 {
                            8
                        } else if seg_left >= 4 {
                            4
                        } else {
                            seg_left
                        };

                        let b_pack_ptr = {
                            let panel_b = cache.get_or_create_panel_b(kc_curr, seg_n);
                            panel_b.pack_from_matrix(
                                b_ptr.add(pc * n + jc + jr + seg_offset),
                                n,
                                kc_curr,
                                seg_n,
                            );
                            panel_b.as_ptr()
                        };
                        let b_pack_aligned = (b_pack_ptr as usize).is_multiple_of(32);

                        let mut ic = 0usize;
                        while ic < m {
                            let block_m = (m - ic).min(mc_block);

                            // Double-buffer A micro-panels within this seg
                            let mut a_buf0 = PackedPanelA::new(mr, kc_curr);
                            let mut a_buf1 = PackedPanelA::new(mr, kc_curr);
                            let mut use_buf0 = true;

                            let mut i_block = 0usize;
                            // Pre-pack first
                            if block_m > 0 {
                                let valid_m0 = block_m.min(mr);
                                let a_src0 = a_ptr.add(ic * k + pc);
                                a_buf0.pack_from_matrix(a_src0, k, valid_m0, kc_curr);
                            }

                            while i_block < block_m {
                                let remain_m = block_m - i_block;
                                let valid_m = remain_m.min(mr);
                                let c_micro = c_ptr.add((ic + i_block) * n + jc + jr + seg_offset);
                                let accumulate = pc != 0;

                                // Pre-pack next in the other buffer
                                let has_next = i_block + mr < block_m;
                                if has_next {
                                    let next_valid_m = (block_m - (i_block + mr)).min(mr);
                                    let a_src_next = a_ptr.add((ic + i_block + mr) * k + pc);
                                    if use_buf0 {
                                        a_buf1.pack_from_matrix(
                                            a_src_next,
                                            k,
                                            next_valid_m,
                                            kc_curr,
                                        );
                                    } else {
                                        a_buf0.pack_from_matrix(
                                            a_src_next,
                                            k,
                                            next_valid_m,
                                            kc_curr,
                                        );
                                    }
                                }

                                let a_micro = if use_buf0 {
                                    a_buf0.as_ptr()
                                } else {
                                    a_buf1.as_ptr()
                                };

                                if seg_n == 16 {
                                    // 6x16
                                    let c_aligned = (c_micro as usize).is_multiple_of(32)
                                        && n.is_multiple_of(8)
                                        && (jr + seg_offset).is_multiple_of(8);
                                    microkernel_6x16_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        c_micro,
                                        kc_curr,
                                        n,
                                        valid_m,
                                        accumulate,
                                        b_pack_aligned,
                                        c_aligned,
                                    );
                                } else if seg_n == 12 {
                                    // 6x12
                                    let c_aligned32 = (c_micro as usize).is_multiple_of(32)
                                        && n.is_multiple_of(8)
                                        && (jr + seg_offset).is_multiple_of(8);
                                    microkernel_6x12_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        c_micro,
                                        kc_curr,
                                        n,
                                        valid_m,
                                        accumulate,
                                        b_pack_aligned,
                                        c_aligned32,
                                    );
                                } else if seg_n == 8 {
                                    // 6x8 with aligned stores when possible
                                    let c_aligned = (c_micro as usize).is_multiple_of(32)
                                        && n.is_multiple_of(8)
                                        && (jr + seg_offset).is_multiple_of(8);
                                    if c_aligned {
                                        microkernel_6x8_packed_aligned(
                                            a_micro, b_pack_ptr, c_micro, kc_curr, n, valid_m, 8,
                                            accumulate,
                                        );
                                    } else {
                                        microkernel_6x8_packed(
                                            a_micro, b_pack_ptr, c_micro, kc_curr, n, valid_m, 8,
                                            accumulate,
                                        );
                                    }
                                } else if seg_n == 4 {
                                    // 6x4
                                    let c_aligned16 = (c_micro as usize).is_multiple_of(16)
                                        && n.is_multiple_of(4)
                                        && (jr + seg_offset).is_multiple_of(4);
                                    microkernel_6x4_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        c_micro,
                                        kc_curr,
                                        n,
                                        valid_m,
                                        accumulate,
                                        b_pack_aligned, // 32-aligned implies 16-aligned
                                        c_aligned16,
                                    );
                                } else {
                                    // 1..3 scalar tail
                                    microkernel_6x_tail_scalar(
                                        a_micro, b_pack_ptr, c_micro, kc_curr, n, valid_m, seg_n,
                                        0, accumulate,
                                    );
                                }

                                use_buf0 = !use_buf0;
                                i_block += mr;
                            }
                            ic += mc_block;
                        }

                        seg_offset += seg_n;
                    }
                });

                jr += remain_n;
            }

            jc += nc_curr;
        }

        pc += kc_block;
    }
}

// ==========================
// Packed GEMM (strided sources)
// ==========================

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_strided_avx2_unaligned(
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
    packed_gemm_avx2_strided(
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

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn mat_mat_strided_avx2_aligned(
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
    // The internal micro-kernels select aligned stores opportunistically; reuse same path
    packed_gemm_avx2_strided(
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

#[inline]
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn packed_gemm_avx2_strided(
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
    let params = BlockingParams::get_cached();
    let kc_block = params.kc;
    let mc_block = params.mc;
    let nr = 16usize; // AVX2
    let mr = 6usize; // AVX2
    let nc_block = 256usize; // N-tiling for better locality

    let mut pc = 0usize;
    while pc < k {
        let kc_curr = (k - pc).min(kc_block);

        let mut jc = 0usize;
        while jc < n {
            let nc_curr = (n - jc).min(nc_block);

            let mut jr = 0usize;
            while jr < nc_curr {
                let block_n = (nc_curr - jr).min(nr);

                with_panel_cache(|cache| {
                    // Segment block_n into {16, 8, 4} chunks to minimize scalar tails
                    let mut seg_offset = 0usize;
                    while seg_offset < block_n {
                        let seg_left = block_n - seg_offset;
                        let seg_n = if seg_left >= 16 {
                            16
                        } else if seg_left >= 12 {
                            12
                        } else if seg_left >= 8 {
                            8
                        } else if seg_left >= 4 {
                            4
                        } else {
                            seg_left
                        };

                        let b_pack_ptr = {
                            let panel_b = cache.get_or_create_panel_b(kc_curr, seg_n);
                            panel_b.pack_from_matrix_strided(
                                b_ptr
                                    .add(pc * b_row_stride + (jc + jr + seg_offset) * b_col_stride),
                                b_row_stride,
                                b_col_stride,
                                kc_curr,
                                seg_n,
                            );
                            panel_b.as_ptr()
                        };
                        let b_pack_aligned = (b_pack_ptr as usize).is_multiple_of(32);

                        let mut ic = 0usize;
                        while ic < m {
                            let block_m = (m - ic).min(mc_block);

                            let a_pack_ptr = {
                                let panel_a = cache.get_or_create_panel_a(block_m, kc_curr);
                                panel_a.pack_from_matrix_strided(
                                    a_ptr.add(ic * a_row_stride + pc * a_col_stride),
                                    a_row_stride,
                                    a_col_stride,
                                    block_m,
                                    kc_curr,
                                );
                                panel_a.as_ptr()
                            };

                            let mut i_block = 0usize;
                            while i_block < block_m {
                                let remain_m = block_m - i_block;
                                let valid_m = remain_m.min(mr);
                                let a_micro = a_pack_ptr.add((i_block / mr) * (mr * kc_curr));
                                let c_base = c_ptr.add(
                                    (ic + i_block) * c_row_stride
                                        + (jc + jr + seg_offset) * c_col_stride,
                                );
                                let accumulate_to_dest = pc != 0;

                                // Use aligned stack tiles to enable aligned stores where possible
                                #[repr(align(32))]
                                struct Tile16([f32; 6 * 16]);

                                if seg_n == 16 {
                                    let mut tile = Tile16([0.0; 6 * 16]);
                                    let tile_ptr = tile.0.as_mut_ptr();
                                    microkernel_6x16_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        tile_ptr,
                                        kc_curr,
                                        16,
                                        valid_m,
                                        false,
                                        b_pack_aligned,
                                        true, // aligned tile
                                    );
                                    for row in 0..valid_m {
                                        let crow = c_base.add(row * c_row_stride);
                                        let src = tile_ptr.add(row * 16);
                                        for col in 0..16 {
                                            let dst = crow.add(col * c_col_stride);
                                            let val = *src.add(col);
                                            if accumulate_to_dest {
                                                *dst += val;
                                            } else {
                                                *dst = val;
                                            }
                                        }
                                    }
                                } else if seg_n == 12 {
                                    let mut tile = Tile16([0.0; 6 * 16]);
                                    let tile_ptr = tile.0.as_mut_ptr();
                                    microkernel_6x12_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        tile_ptr,
                                        kc_curr,
                                        16,
                                        valid_m,
                                        false,
                                        b_pack_aligned,
                                        true,
                                    );
                                    for row in 0..valid_m {
                                        let crow = c_base.add(row * c_row_stride);
                                        let src = tile_ptr.add(row * 16);
                                        for col in 0..12 {
                                            let dst = crow.add(col * c_col_stride);
                                            let val = *src.add(col);
                                            if accumulate_to_dest {
                                                *dst += val;
                                            } else {
                                                *dst = val;
                                            }
                                        }
                                    }
                                } else if seg_n == 8 {
                                    let mut tile = Tile16([0.0; 6 * 16]);
                                    let tile_ptr = tile.0.as_mut_ptr();
                                    microkernel_6x8_packed(
                                        a_micro, b_pack_ptr, tile_ptr, kc_curr, 8, valid_m, 8,
                                        false,
                                    );
                                    for row in 0..valid_m {
                                        let crow = c_base.add(row * c_row_stride);
                                        let src = tile_ptr.add(row * 16);
                                        for col in 0..8 {
                                            let dst = crow.add(col * c_col_stride);
                                            let val = *src.add(col);
                                            if accumulate_to_dest {
                                                *dst += val;
                                            } else {
                                                *dst = val;
                                            }
                                        }
                                    }
                                } else if seg_n == 4 {
                                    let mut tile = Tile16([0.0; 6 * 16]);
                                    let tile_ptr = tile.0.as_mut_ptr();
                                    // Compute into the 4-wide tile region and scatter
                                    microkernel_6x4_packed(
                                        a_micro,
                                        b_pack_ptr,
                                        tile_ptr,
                                        kc_curr,
                                        4,
                                        valid_m,
                                        false,
                                        b_pack_aligned,
                                        true,
                                    );
                                    for row in 0..valid_m {
                                        let crow = c_base.add(row * c_row_stride);
                                        let src = tile_ptr.add(row * 16);
                                        for col in 0..4 {
                                            let dst = crow.add(col * c_col_stride);
                                            let val = *src.add(col);
                                            if accumulate_to_dest {
                                                *dst += val;
                                            } else {
                                                *dst = val;
                                            }
                                        }
                                    }
                                } else {
                                    // 1..3 columns: compute into tile then scatter
                                    let mut tile = Tile16([0.0; 6 * 16]);
                                    let tile_ptr = tile.0.as_mut_ptr();
                                    microkernel_6x_tail_scalar(
                                        a_micro, b_pack_ptr, tile_ptr, kc_curr, seg_n, valid_m,
                                        seg_n, 0, false,
                                    );
                                    for row in 0..valid_m {
                                        let crow = c_base.add(row * c_row_stride);
                                        let src = tile_ptr.add(row * 16);
                                        for col in 0..seg_n {
                                            let dst = crow.add(col * c_col_stride);
                                            let val = *src.add(col);
                                            if accumulate_to_dest {
                                                *dst += val;
                                            } else {
                                                *dst = val;
                                            }
                                        }
                                    }
                                }

                                i_block += mr;
                            }
                            ic += mc_block;
                        }

                        seg_offset += seg_n;
                    }
                });

                jr += block_n;
            }

            jc += nc_curr;
        }

        pc += kc_block;
    }
}
