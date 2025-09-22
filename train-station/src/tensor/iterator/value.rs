//! High-performance value iterators over tensor data

use crate::tensor::core::Tensor;
use crate::tensor::iterator::collect::optimized_copy;
use std::iter::FusedIterator;
use std::marker::PhantomData;

/// Read-only iterator over tensor element values (by value)
///
/// - Contiguous tensors: zero-copy over underlying memory
/// - Non-contiguous tensors: materializes a contiguous view once, then iterates
pub struct TensorValuesIter<'a> {
    ptr: *const f32,
    len: usize,
    pos: usize,
    // Keep an owned contiguous buffer alive for non-contiguous tensors
    #[allow(dead_code)]
    owner: Option<Tensor>,
    _marker: PhantomData<&'a f32>,
}

impl<'a> TensorValuesIter<'a> {
    #[inline]
    pub(crate) fn new(t: &'a Tensor) -> Self {
        // Treat any zero-stride view as non-contiguous to avoid unsafe flat iteration
        let has_zero_stride = t.strides().contains(&0);
        if (t.is_contiguous() && !has_zero_stride) || t.size() == 0 {
            let len = t.size();
            let ptr = unsafe { t.as_ptr() };
            Self {
                ptr,
                len,
                pos: 0,
                owner: None,
                _marker: PhantomData,
            }
        } else {
            // Materialize once into a contiguous, aligned buffer WITHOUT grad wiring
            let len = t.size();
            let mut owned = Tensor::new_uninitialized(vec![len]);
            unsafe {
                copy_strided_to_contiguous_1d(t, owned.as_mut_ptr());
            }
            let ptr = unsafe { owned.as_ptr() };
            Self {
                ptr,
                len,
                pos: 0,
                owner: Some(owned),
                _marker: PhantomData,
            }
        }
    }
}

impl<'a> Iterator for TensorValuesIter<'a> {
    type Item = f32;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.len {
            return None;
        }
        let i = self.pos;
        self.pos += 1;
        unsafe { Some(*self.ptr.add(i)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len.saturating_sub(self.pos);
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorValuesIter<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.len.saturating_sub(self.pos)
    }
}

impl<'a> FusedIterator for TensorValuesIter<'a> {}

impl Tensor {
    /// Iterate over tensor values efficiently (by value)
    ///
    /// - Contiguous tensors: zero-copy over the underlying memory
    /// - Non-contiguous tensors: materializes a contiguous view once, then iterates
    ///
    /// Gradient tracking note: This iterator yields values (f32) and does not
    /// carry per-element gradient semantics. Use whole-tensor ops for autograd.
    #[inline]
    pub fn iter_values(&self) -> TensorValuesIter<'_> {
        TensorValuesIter::new(self)
    }

    /// Iterate mutably over tensor values (contiguous only)
    ///
    /// This returns a standard slice iterator over the underlying memory.
    ///
    /// Panics if the tensor is not contiguous. Use `contiguous()` beforehand
    /// to materialize a contiguous buffer if needed.
    #[inline]
    pub fn iter_values_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        assert!(
            self.is_contiguous(),
            "iter_values_mut requires contiguous tensor; call contiguous() first"
        );
        self.data_mut().iter_mut()
    }
}

/// Copy an arbitrary-strided tensor into a 1D contiguous buffer (aligned)
#[inline]
unsafe fn copy_strided_to_contiguous_1d(src: &Tensor, dst_ptr: *mut f32) {
    let size = src.size();
    if size == 0 {
        return;
    }
    let rank = src.shape().rank();
    let src_base = src.as_ptr();

    if rank >= 1 && src.stride(rank - 1) == 1 {
        // Fast row-by-row path: last dim contiguous
        let dims = src.shape().dims();
        let row_len = dims[rank - 1];
        let outer: usize = if rank == 1 {
            1
        } else {
            dims[..rank - 1].iter().product()
        };
        let strides = src.strides();
        let mut coords = vec![0usize; rank];
        for outer_idx in 0..outer {
            if rank > 1 {
                let mut tmp = outer_idx;
                for i in (0..rank - 1).rev() {
                    let d = dims[i];
                    coords[i] = if d == 0 { 0 } else { tmp % d };
                    if d != 0 {
                        tmp /= d;
                    }
                }
            }
            coords[rank - 1] = 0;
            let mut src_off = 0usize;
            for i in 0..rank {
                src_off += coords[i] * strides[i];
            }
            // destination linear row index
            let mut dst_row_index = 0usize;
            if rank > 1 {
                for i in 0..rank - 1 {
                    dst_row_index = dst_row_index * dims[i] + coords[i];
                }
            }
            let dst_off = dst_row_index * row_len;
            optimized_copy(src_base.add(src_off), dst_ptr.add(dst_off), row_len);
        }
        return;
    }

    // General fallback: compute coordinates for each element
    let dims = src.shape().dims();
    for dst_idx in 0..size {
        let mut coords = vec![0usize; rank];
        let mut tmp = dst_idx;
        for i in (0..rank).rev() {
            let d = dims[i];
            coords[i] = if d == 0 { 0 } else { tmp % d };
            if d != 0 {
                tmp /= d;
            }
        }
        let src_off = src.memory_offset(&coords);
        *dst_ptr.add(dst_idx) = *src_base.add(src_off);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_values_contiguous() {
        let t = Tensor::from_slice(&(0..8).map(|i| i as f32).collect::<Vec<_>>(), vec![8]).unwrap();
        let vals: Vec<f32> = t.iter_values().collect();
        assert_eq!(vals, (0..8).map(|i| i as f32).collect::<Vec<_>>());
    }

    #[test]
    fn test_iter_values_mut_contiguous() {
        let mut t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        for v in t.iter_values_mut() {
            *v *= 2.0;
        }
        assert_eq!(t.data(), &[2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    #[should_panic]
    fn test_iter_values_mut_panics_non_contiguous() {
        let base =
            Tensor::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), vec![2, 3]).unwrap();
        let perm = base.permute(vec![1, 0]);
        let mut owned = perm.contiguous();
        // Now owned is contiguous; but the panic test requires non-contiguous, so use perm directly
        let mut not_contig = perm; // non-contiguous view
        let _ = not_contig.iter_values_mut();
        // Should panic before this point
        let _ = owned.iter_values_mut(); // ensure method compiles
    }
}
