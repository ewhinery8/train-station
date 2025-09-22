//! Scalar matrix multiplication kernels with optimized loop unrolling

#![allow(clippy::too_many_arguments)]

// Use Kahan/Neumaier style only for sufficiently large K to avoid overhead on small reductions
const KAHAN_LARGE_K_THRESHOLD: usize = 128;

type DotFn = unsafe fn(*const f32, *const f32, usize, usize, usize) -> f32;

#[inline(always)]
unsafe fn dot_plain_unrolled8(
    a_ptr: *const f32,
    b_ptr: *const f32,
    size: usize,
    a_stride: usize,
    b_stride: usize,
) -> f32 {
    let mut acc0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut a_off = 0usize;
    let mut b_off = 0usize;

    let unroll = 8usize;
    let main = size / unroll * unroll;
    let mut i = 0usize;
    while i < main {
        // 4 to acc0
        acc0 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc0 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc0 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc0 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        // 4 to acc1
        acc1 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc1 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc1 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        acc1 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;

        i += unroll;
    }

    // Tail
    while i < size {
        acc0 += *a_ptr.add(a_off) * *b_ptr.add(b_off);
        a_off += a_stride;
        b_off += b_stride;
        i += 1;
    }

    acc0 + acc1
}

#[inline(always)]
unsafe fn dot_kahan2_unrolled8(
    a_ptr: *const f32,
    b_ptr: *const f32,
    size: usize,
    a_stride: usize,
    b_stride: usize,
) -> f32 {
    let mut acc0 = 0.0f32;
    let mut c0 = 0.0f32;
    let mut acc1 = 0.0f32;
    let mut c1 = 0.0f32;
    let mut a_off = 0usize;
    let mut b_off = 0usize;

    let unroll = 8usize;
    let main = size / unroll * unroll;
    let mut i = 0usize;
    while i < main {
        // 4 updates into (acc0, c0)
        let p0 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y0 = p0 - c0;
        let t0 = acc0 + y0;
        c0 = (t0 - acc0) - y0;
        acc0 = t0;
        a_off += a_stride;
        b_off += b_stride;

        let p1 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y1 = p1 - c0;
        let t1 = acc0 + y1;
        c0 = (t1 - acc0) - y1;
        acc0 = t1;
        a_off += a_stride;
        b_off += b_stride;

        let p2 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y2 = p2 - c0;
        let t2 = acc0 + y2;
        c0 = (t2 - acc0) - y2;
        acc0 = t2;
        a_off += a_stride;
        b_off += b_stride;

        let p3 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y3 = p3 - c0;
        let t3 = acc0 + y3;
        c0 = (t3 - acc0) - y3;
        acc0 = t3;
        a_off += a_stride;
        b_off += b_stride;

        // 4 updates into (acc1, c1)
        let p4 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y4 = p4 - c1;
        let t4 = acc1 + y4;
        c1 = (t4 - acc1) - y4;
        acc1 = t4;
        a_off += a_stride;
        b_off += b_stride;

        let p5 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y5 = p5 - c1;
        let t5 = acc1 + y5;
        c1 = (t5 - acc1) - y5;
        acc1 = t5;
        a_off += a_stride;
        b_off += b_stride;

        let p6 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y6 = p6 - c1;
        let t6 = acc1 + y6;
        c1 = (t6 - acc1) - y6;
        acc1 = t6;
        a_off += a_stride;
        b_off += b_stride;

        let p7 = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y7 = p7 - c1;
        let t7 = acc1 + y7;
        c1 = (t7 - acc1) - y7;
        acc1 = t7;
        a_off += a_stride;
        b_off += b_stride;

        i += unroll;
    }

    // Merge the two compensated sums
    let mut sum = (acc0 + c0) + (acc1 + c1);
    let mut c_tail = 0.0f32;

    // Tail with Neumaier-style compensation folded into the merged sum
    while i < size {
        let p = *a_ptr.add(a_off) * *b_ptr.add(b_off);
        let y = p - c_tail;
        let t = sum + y;
        c_tail = (t - sum) - y;
        sum = t;
        a_off += a_stride;
        b_off += b_stride;
        i += 1;
    }

    sum
}

/// Optimized 1D @ 1D dot product (scalar result)
#[inline(always)]
pub unsafe fn matmul_scalar_1d_1d(
    a_ptr: *const f32,
    b_ptr: *const f32,
    size: usize,
    a_stride: usize,
    b_stride: usize,
) -> f32 {
    let dot: DotFn = if size >= KAHAN_LARGE_K_THRESHOLD {
        dot_kahan2_unrolled8
    } else {
        dot_plain_unrolled8
    };
    dot(a_ptr, b_ptr, size, a_stride, b_stride)
}

/// Optimized 1D @ 2D vector-matrix multiplication (v^T * M)
#[inline(always)]
pub unsafe fn matmul_scalar_1d_2d(
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
    let dot: DotFn = if k >= KAHAN_LARGE_K_THRESHOLD {
        dot_kahan2_unrolled8
    } else {
        dot_plain_unrolled8
    };

    let mut c_off = 0usize;
    for j in 0..n {
        let a_col_start = a_ptr; // vector broadcasted over columns
        let b_col_start = b_ptr.add(j * b_col_stride);
        let sum = dot(a_col_start, b_col_start, k, a_stride, b_row_stride);
        *c_ptr.add(c_off) = sum;
        c_off += c_stride;
    }
}

/// Optimized 2D @ 1D matrix-vector multiplication (M * v)
#[inline(always)]
pub unsafe fn matmul_scalar_2d_1d(
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
    let dot: DotFn = if k >= KAHAN_LARGE_K_THRESHOLD {
        dot_kahan2_unrolled8
    } else {
        dot_plain_unrolled8
    };

    let mut c_off = 0usize;
    for i in 0..m {
        let a_row_start = a_ptr.add(i * a_row_stride);
        let b_start = b_ptr;
        let sum = dot(a_row_start, b_start, k, a_col_stride, b_stride);
        *c_ptr.add(c_off) = sum;
        c_off += c_stride;
    }
}

/// Optimized 2D @ 2D matrix-matrix multiplication
#[inline(always)]
pub unsafe fn matmul_scalar_2d_2d(
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
    let dot: DotFn = if k >= KAHAN_LARGE_K_THRESHOLD {
        dot_kahan2_unrolled8
    } else {
        dot_plain_unrolled8
    };

    for i in 0..m {
        let a_row_start = a_ptr.add(i * a_row_stride);
        let c_row_start = c_ptr.add(i * c_row_stride);

        for j in 0..n {
            let b_col_start = b_ptr.add(j * b_col_stride);
            let sum = dot(a_row_start, b_col_start, k, a_col_stride, b_row_stride);
            *c_row_start.add(j * c_col_stride) = sum;
        }
    }
}
