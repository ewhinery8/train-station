//! Collection helpers for iterators yielding tensors

use crate::gradtrack::is_grad_enabled;
#[cfg(target_arch = "x86_64")]
use crate::tensor::core::memory::{detect_runtime_simd, simd_alignment_bytes, SimdLevel};
use crate::tensor::core::Tensor;
use std::iter::FromIterator;

impl Tensor {
    /// Collect tensors into a single tensor with target shape, copying data in iterator order.
    /// Optimizes copy using SIMD when available; asserts total size matches.
    #[inline]
    pub fn collect_into_shape<I: IntoIterator<Item = Tensor>>(iter: I, dims: Vec<usize>) -> Tensor {
        let total: usize = dims.iter().copied().product();
        let elements: Vec<Tensor> = iter.into_iter().collect();
        let sum_sizes: usize = elements.iter().map(|t| t.size()).sum();
        assert_eq!(
            sum_sizes, total,
            "collect_into_shape: element sizes {} do not match target size {}",
            sum_sizes, total
        );
        let requires_grad = elements.iter().any(|t| t.requires_grad()) && is_grad_enabled();

        if requires_grad {
            // Autograd-preserving path: flatten each element, concatenate with cat,
            // then reshape to the requested dims via view/reshape to preserve GradFn wiring.
            let mut flat_parts: Vec<Tensor> = Vec::with_capacity(elements.len());
            for t in elements.into_iter() {
                // Ensure rank 1 for cat
                flat_parts.push(t.flatten());
            }
            let concatenated = Tensor::cat(&flat_parts, 0); // [total]
                                                            // Use view for reshape (registers GradFn::Reshape)
            let new_shape: Vec<i32> = dims.iter().map(|&d| d as i32).collect();
            let out = concatenated.view(new_shape);
            return out;
        }

        // Fast forward-only copy when no gradients are required
        let mut result = Tensor::new_uninitialized(dims);
        let mut offset = 0usize;
        unsafe {
            let dst = result.as_mut_ptr();
            for t in &elements {
                let sz = t.size();
                if sz == 0 {
                    continue;
                }
                optimized_copy(t.as_ptr(), dst.add(offset), sz);
                offset += sz;
            }
        }
        result
    }
}

// ===== Inherent convenience methods on core iterators =====
// These allow calling `.collect_shape(dims)` directly on the iterator returned by
// Tensor's iterator constructors without importing any extension traits. For more
// complex iterator chains that use adapters like `.map(...)`, the extension traits
// are still available (and re-exported at crate root) to enable method-style usage.

use crate::tensor::iterator::chunks::{TensorChunksExactIterator, TensorChunksIterator};
use crate::tensor::iterator::element::TensorElementIterator;
use crate::tensor::iterator::viewdim::TensorDimIterator;
use crate::tensor::iterator::windows::TensorWindowsIterator;

impl<'a> TensorChunksIterator<'a> {
    /// Collect this chunks iterator into a tensor with the provided shape.
    ///
    /// Gradient tracking: If any produced chunk tensor requires gradients and
    /// gradients are enabled, the resulting tensor will preserve autograd
    /// connections back to the original source tensor.
    #[inline]
    pub fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

impl<'a> TensorChunksExactIterator<'a> {
    /// Collect this exact-sized chunks iterator into a tensor with the provided shape.
    /// See [`TensorChunksIterator::collect_shape`] for gradient behavior.
    #[inline]
    pub fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

impl<'a> TensorWindowsIterator<'a> {
    /// Collect this windows iterator into a tensor with the provided shape.
    /// See [`TensorChunksIterator::collect_shape`] for gradient behavior.
    #[inline]
    pub fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

impl<'a> TensorDimIterator<'a> {
    /// Collect this dimension iterator into a tensor with the provided shape.
    /// See [`TensorChunksIterator::collect_shape`] for gradient behavior.
    #[inline]
    pub fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

impl<'a> TensorElementIterator<'a> {
    /// Collect this element iterator into a tensor with the provided shape.
    /// See [`TensorChunksIterator::collect_shape`] for gradient behavior.
    #[inline]
    pub fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

impl Tensor {
    /// Inherent helper to collect any iterator of `Tensor` into the specified shape.
    ///
    /// This is equivalent to calling `.collect_shape(dims)` on the iterator via the
    /// extension trait, but provided here as a convenience that works without bringing
    /// the trait into scope. Example:
    ///
    /// ```
    /// # use train_station::Tensor;
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    /// let y = Tensor::collect_shape_from(x.iter_elements().map(|e| e.mul_scalar(2.0)), vec![4]);
    /// ```
    ///
    /// Gradient tracking behavior matches [`Tensor::collect_into_shape`].
    #[inline]
    pub fn collect_shape_from<I: IntoIterator<Item = Tensor>>(iter: I, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(iter, dims)
    }

    /// Inherent helper to collect any iterator of `f32` into a shaped tensor.
    /// This mirrors the `ValuesCollectExt::collect_shape` functionality but does
    /// not require importing the extension trait.
    #[inline]
    pub fn collect_values_shape<I: IntoIterator<Item = f32>>(iter: I, dims: Vec<usize>) -> Tensor {
        let total: usize = dims.iter().copied().product();
        let mut out = Tensor::new_uninitialized(dims);
        if total == 0 {
            return out;
        }
        unsafe {
            let dst = out.as_mut_ptr();
            let mut i = 0usize;
            for v in iter {
                if i >= total {
                    break;
                }
                *dst.add(i) = v;
                i += 1;
            }
            assert_eq!(
                i, total,
                "values collect_shape: provided iterator produced {} values, expected {}",
                i, total
            );
        }
        out
    }
}

// Optimized collection from Iterator<Item=f32>
impl FromIterator<f32> for Tensor {
    /// Collect f32 values into a 1D contiguous, SIMD-aligned Tensor
    ///
    /// - Pre-allocates with optimized alignment and possible padding
    /// - Copies with AVX512→AVX2→SSE→scalar fallback
    /// - No gradient tracking is set on the result
    #[inline]
    fn from_iter<I: IntoIterator<Item = f32>>(iter: I) -> Self {
        // First pass: collect into a Vec<f32> to know exact length
        // Note: we could attempt a multi-pass size_hint growth strategy, but
        // a single Vec collect is generally fastest in practice and keeps code simple.
        let v: Vec<f32> = iter.into_iter().collect();
        let n = v.len();
        if n == 0 {
            return Tensor::new(vec![0]);
        }

        let mut out = Tensor::new_uninitialized(vec![n]);
        unsafe {
            optimized_copy(v.as_ptr(), out.as_mut_ptr(), n);
        }
        out
    }
}

/// Use SIMD-optimized copy when available; falls back to scalar/unrolled copies.
#[inline]
pub(crate) unsafe fn optimized_copy(src: *const f32, dst: *mut f32, count: usize) {
    if count == 0 {
        return;
    }
    if count <= 32 {
        std::ptr::copy_nonoverlapping(src, dst, count);
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        match detect_runtime_simd() {
            SimdLevel::Avx512 => {
                if simd_copy_avx512_best(src, dst, count) {
                    return;
                }
            }
            SimdLevel::Avx2 => {
                if simd_copy_avx2_best(src, dst, count) {
                    return;
                }
            }
            SimdLevel::Sse2 => {
                if simd_copy_sse_best(src, dst, count) {
                    return;
                }
            }
            SimdLevel::Scalar => {}
        }
    }

    scalar_copy_unrolled(src, dst, count);
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn simd_copy_avx512_best(src: *const f32, dst: *mut f32, count: usize) -> bool {
    if !is_x86_feature_detected!("avx512f") || count < 16 {
        return false;
    }
    let align = simd_alignment_bytes(SimdLevel::Avx512);
    let src_mod = (src as usize) % align;
    let dst_mod = (dst as usize) % align;
    let src_al = src_mod == 0;
    let dst_al = dst_mod == 0;
    if src_al && dst_al {
        simd_copy_avx512_aligned(src, dst, count);
    } else if src_mod == dst_mod {
        let bytes_to_align = if src_mod == 0 { 0 } else { align - src_mod };
        let elems_to_align = (bytes_to_align / std::mem::size_of::<f32>()).min(count);
        if elems_to_align > 0 && elems_to_align < count {
            std::ptr::copy_nonoverlapping(src, dst, elems_to_align);
            let src2 = src.add(elems_to_align);
            let dst2 = dst.add(elems_to_align);
            let rem = count - elems_to_align;
            simd_copy_avx512_aligned(src2, dst2, rem);
        } else {
            simd_copy_avx512_unaligned(src, dst, count);
        }
    } else {
        simd_copy_avx512_unaligned(src, dst, count);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn simd_copy_avx512_aligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 64usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let a = _mm512_load_ps(src.add(offset));
        let b = _mm512_load_ps(src.add(offset + 16));
        let c = _mm512_load_ps(src.add(offset + 32));
        let d = _mm512_load_ps(src.add(offset + 48));
        _mm512_store_ps(dst.add(offset), a);
        _mm512_store_ps(dst.add(offset + 16), b);
        _mm512_store_ps(dst.add(offset + 32), c);
        _mm512_store_ps(dst.add(offset + 48), d);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 16 {
        let v = _mm512_load_ps(src.add(offset));
        _mm512_store_ps(dst.add(offset), v);
        offset += 16;
        rem -= 16;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx512f")]
unsafe fn simd_copy_avx512_unaligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 64usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let a = _mm512_loadu_ps(src.add(offset));
        let b = _mm512_loadu_ps(src.add(offset + 16));
        let c = _mm512_loadu_ps(src.add(offset + 32));
        let d = _mm512_loadu_ps(src.add(offset + 48));
        _mm512_storeu_ps(dst.add(offset), a);
        _mm512_storeu_ps(dst.add(offset + 16), b);
        _mm512_storeu_ps(dst.add(offset + 32), c);
        _mm512_storeu_ps(dst.add(offset + 48), d);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 16 {
        let v = _mm512_loadu_ps(src.add(offset));
        _mm512_storeu_ps(dst.add(offset), v);
        offset += 16;
        rem -= 16;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn simd_copy_avx2_best(src: *const f32, dst: *mut f32, count: usize) -> bool {
    if !is_x86_feature_detected!("avx2") || count < 8 {
        return false;
    }
    let align = simd_alignment_bytes(SimdLevel::Avx2);
    let src_mod = (src as usize) % align;
    let dst_mod = (dst as usize) % align;
    let src_al = src_mod == 0;
    let dst_al = dst_mod == 0;
    if src_al && dst_al {
        simd_copy_avx2_aligned(src, dst, count);
    } else if src_mod == dst_mod {
        let bytes_to_align = if src_mod == 0 { 0 } else { align - src_mod };
        let elems_to_align = (bytes_to_align / std::mem::size_of::<f32>()).min(count);
        if elems_to_align > 0 && elems_to_align < count {
            std::ptr::copy_nonoverlapping(src, dst, elems_to_align);
            let src2 = src.add(elems_to_align);
            let dst2 = dst.add(elems_to_align);
            let rem = count - elems_to_align;
            simd_copy_avx2_aligned(src2, dst2, rem);
        } else {
            simd_copy_avx2_unaligned(src, dst, count);
        }
    } else {
        simd_copy_avx2_unaligned(src, dst, count);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn simd_copy_avx2_aligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 32usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let v1 = _mm256_load_ps(src.add(offset));
        let v2 = _mm256_load_ps(src.add(offset + 8));
        let v3 = _mm256_load_ps(src.add(offset + 16));
        let v4 = _mm256_load_ps(src.add(offset + 24));
        _mm256_store_ps(dst.add(offset), v1);
        _mm256_store_ps(dst.add(offset + 8), v2);
        _mm256_store_ps(dst.add(offset + 16), v3);
        _mm256_store_ps(dst.add(offset + 24), v4);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 8 {
        let v = _mm256_load_ps(src.add(offset));
        _mm256_store_ps(dst.add(offset), v);
        offset += 8;
        rem -= 8;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn simd_copy_avx2_unaligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 32usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let v1 = _mm256_loadu_ps(src.add(offset));
        let v2 = _mm256_loadu_ps(src.add(offset + 8));
        let v3 = _mm256_loadu_ps(src.add(offset + 16));
        let v4 = _mm256_loadu_ps(src.add(offset + 24));
        _mm256_storeu_ps(dst.add(offset), v1);
        _mm256_storeu_ps(dst.add(offset + 8), v2);
        _mm256_storeu_ps(dst.add(offset + 16), v3);
        _mm256_storeu_ps(dst.add(offset + 24), v4);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 8 {
        let v = _mm256_loadu_ps(src.add(offset));
        _mm256_storeu_ps(dst.add(offset), v);
        offset += 8;
        rem -= 8;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn simd_copy_sse_best(src: *const f32, dst: *mut f32, count: usize) -> bool {
    if !is_x86_feature_detected!("sse2") || count < 4 {
        return false;
    }
    let align = simd_alignment_bytes(SimdLevel::Sse2);
    let src_mod = (src as usize) % align;
    let dst_mod = (dst as usize) % align;
    let src_al = src_mod == 0;
    let dst_al = dst_mod == 0;
    if src_al && dst_al {
        simd_copy_sse_aligned(src, dst, count);
    } else if src_mod == dst_mod {
        let bytes_to_align = if src_mod == 0 { 0 } else { align - src_mod };
        let elems_to_align = (bytes_to_align / std::mem::size_of::<f32>()).min(count);
        if elems_to_align > 0 && elems_to_align < count {
            std::ptr::copy_nonoverlapping(src, dst, elems_to_align);
            let src2 = src.add(elems_to_align);
            let dst2 = dst.add(elems_to_align);
            let rem = count - elems_to_align;
            simd_copy_sse_aligned(src2, dst2, rem);
        } else {
            simd_copy_sse_unaligned(src, dst, count);
        }
    } else {
        simd_copy_sse_unaligned(src, dst, count);
    }
    true
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn simd_copy_sse_aligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 16usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let a = _mm_load_ps(src.add(offset));
        let b = _mm_load_ps(src.add(offset + 4));
        let c = _mm_load_ps(src.add(offset + 8));
        let d = _mm_load_ps(src.add(offset + 12));
        _mm_store_ps(dst.add(offset), a);
        _mm_store_ps(dst.add(offset + 4), b);
        _mm_store_ps(dst.add(offset + 8), c);
        _mm_store_ps(dst.add(offset + 12), d);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 4 {
        let v = _mm_load_ps(src.add(offset));
        _mm_store_ps(dst.add(offset), v);
        offset += 4;
        rem -= 4;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse2")]
unsafe fn simd_copy_sse_unaligned(src: *const f32, dst: *mut f32, count: usize) {
    use std::arch::x86_64::*;
    let mut offset = 0usize;
    let block = 16usize;
    let n_blocks = count / block;
    for _ in 0..n_blocks {
        let a = _mm_loadu_ps(src.add(offset));
        let b = _mm_loadu_ps(src.add(offset + 4));
        let c = _mm_loadu_ps(src.add(offset + 8));
        let d = _mm_loadu_ps(src.add(offset + 12));
        _mm_storeu_ps(dst.add(offset), a);
        _mm_storeu_ps(dst.add(offset + 4), b);
        _mm_storeu_ps(dst.add(offset + 8), c);
        _mm_storeu_ps(dst.add(offset + 12), d);
        offset += block;
    }
    let mut rem = count - offset;
    while rem >= 4 {
        let v = _mm_loadu_ps(src.add(offset));
        _mm_storeu_ps(dst.add(offset), v);
        offset += 4;
        rem -= 4;
    }
    if rem > 0 {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), rem);
    }
}

#[inline]
unsafe fn scalar_copy_unrolled(src: *const f32, dst: *mut f32, count: usize) {
    let unroll = 8;
    let blocks = count / unroll;
    let mut offset = 0usize;
    for _ in 0..blocks {
        *dst.add(offset) = *src.add(offset);
        *dst.add(offset + 1) = *src.add(offset + 1);
        *dst.add(offset + 2) = *src.add(offset + 2);
        *dst.add(offset + 3) = *src.add(offset + 3);
        *dst.add(offset + 4) = *src.add(offset + 4);
        *dst.add(offset + 5) = *src.add(offset + 5);
        *dst.add(offset + 6) = *src.add(offset + 6);
        *dst.add(offset + 7) = *src.add(offset + 7);
        offset += unroll;
    }
    if offset < count {
        std::ptr::copy_nonoverlapping(src.add(offset), dst.add(offset), count - offset);
    }
}

/// Extension trait to collect iterator of tensors into provided shape.
pub trait TensorCollectExt: Iterator<Item = Tensor> + Sized {
    fn collect_shape(self, dims: Vec<usize>) -> Tensor;
}

impl<I> TensorCollectExt for I
where
    I: Iterator<Item = Tensor> + Sized,
{
    #[inline]
    fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        Tensor::collect_into_shape(self, dims)
    }
}

/// Extension trait to collect Iterator<Item=f32> directly into a shaped Tensor
///
/// This trait is automatically implemented for any `Iterator<Item = f32>`, so you can call
/// `collect_shape()` directly on iterators yielding f32 values without importing the trait.
pub trait ValuesCollectExt: Iterator<Item = f32> + Sized {
    /// Collect f32 values from this iterator into a tensor with the specified shape
    fn collect_shape(self, dims: Vec<usize>) -> Tensor;
}

impl<I> ValuesCollectExt for I
where
    I: Iterator<Item = f32> + Sized,
{
    #[inline]
    fn collect_shape(self, dims: Vec<usize>) -> Tensor {
        let total: usize = dims.iter().copied().product();
        let mut out = Tensor::new_uninitialized(dims);
        if total == 0 {
            return out;
        }

        // For small datasets, use direct iteration to avoid allocation overhead
        if total <= 64 {
            unsafe {
                let dst = out.as_mut_ptr();
                let mut i = 0usize;
                for v in self {
                    if i >= total {
                        break;
                    }
                    *dst.add(i) = v;
                    i += 1;
                }
                assert_eq!(
                    i, total,
                    "values collect_shape: provided iterator produced {} values, expected {}",
                    i, total
                );
            }
            return out;
        }

        // For larger datasets, collect into temporary buffer then use optimized_copy
        let temp_data: Vec<f32> = self.collect();
        assert_eq!(
            temp_data.len(),
            total,
            "values collect_shape: provided iterator produced {} values, expected {}",
            temp_data.len(),
            total
        );

        unsafe {
            optimized_copy(temp_data.as_ptr(), out.as_mut_ptr(), total);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collect_shape() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]).unwrap();
        let mat = t.iter_chunks(2).collect_shape(vec![3, 2]);
        assert_eq!(mat.shape().dims(), &[3, 2]);
        assert_eq!(mat.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_collect_shape_with_grad_preserves_backward() {
        use crate::gradtrack::is_grad_enabled;
        use crate::tensor::core::Tensor;

        if !is_grad_enabled() {
            // Ensure grad is enabled in normal test runs; skip if not
        }

        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();
        // Split in chunks of 2, scale each chunk, collect with target shape [2,2]
        let parts: Vec<Tensor> = t.iter_chunks(2).map(|c| c.mul_scalar(3.0)).collect();
        let y = parts.into_iter().collect_shape(vec![2, 2]);
        assert!(y.requires_grad());

        let mut loss = y.sum();
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        // Each input element appears exactly once in the collected tensor and is scaled by 3
        assert_eq!(g.data(), &[3.0, 3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_collect_from_values_into_tensor() {
        // Collect Iterator<f32> into Tensor via FromIterator<f32>
        let vals = (0..16).map(|i| i as f32);
        let t: Tensor = vals.collect();
        assert_eq!(t.shape().dims(), &[16]);
        assert_eq!(t.data()[0], 0.0);
        assert_eq!(t.data()[15], 15.0);
    }

    #[test]
    fn test_values_iter_then_collect_shape() {
        // values() + collect into a shaped tensor using collect_into_shape
        let base =
            Tensor::from_slice(&(0..12).map(|i| i as f32).collect::<Vec<_>>(), vec![3, 4]).unwrap();
        let flat_vals = base.iter_values();
        let collected: Tensor = flat_vals.collect();
        assert_eq!(collected.shape().dims(), &[12]);
        // reshape using view semantics
        let shaped = collected.view(vec![3, 4]);
        assert_eq!(shaped.shape().dims(), &[3, 4]);
        assert_eq!(shaped.get(&[2, 3]), 11.0);
    }

    #[test]
    fn test_values_collect_shape_direct() {
        // No need to import trait - collect_shape is available on Iterator<Item=f32>
        let shaped: Tensor = (0..12).map(|i| i as f32).collect_shape(vec![3, 4]);
        assert_eq!(shaped.shape().dims(), &[3, 4]);
        assert_eq!(shaped.get(&[0, 0]), 0.0);
        assert_eq!(shaped.get(&[2, 3]), 11.0);
    }
}
