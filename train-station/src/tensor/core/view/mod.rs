//! Core View System: safe, reusable zero-copy views over tensor memory
//!
//! This module provides a small, generic, and safe interface to create views
//! (reshape, transpose, slice, and as_strided) over existing tensor memory.
//! It validates bounds against the allocation capacity (including SIMD padding)
//! and preserves shared ownership via the allocation owner.
//!
//! Design goals:
//! - Safety first: every view validates it stays within the allocation
//! - No copies: zero-copy views by sharing the same `Allocation`
//! - Reusable: concise helpers used by higher-level transform APIs
//! - Future-proof: works with padded capacities and strided layouts

use crate::tensor::core::Tensor;
use crate::tensor::Shape;

/// Errors that can occur while creating views
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ViewError {
    /// The requested reshape is invalid (mismatched element counts or non-contiguous source)
    InvalidReshape {
        requested: Vec<usize>,
        source_size: usize,
        is_contiguous: bool,
    },
    /// The requested transpose permutation is invalid
    InvalidPermutation { rank: usize, perm: Vec<usize> },
    /// Slice parameters are invalid
    InvalidSlice {
        start: usize,
        step: usize,
        length: usize,
    },
    /// The resulting view would access memory out of bounds
    OutOfBounds {
        required_max_index: usize,
        capacity: usize,
    },
}

fn capacity_elems_of(t: &Tensor) -> usize {
    if let Some(owner) = t.allocation_owner() {
        owner.capacity_elems()
    } else {
        // No owner (e.g., zero-sized dangling) – fall back to logical size
        t.size()
    }
}

fn max_index_offset(dims: &[usize], strides: &[usize]) -> usize {
    if dims.is_empty() {
        return 0;
    }
    let mut acc = 0usize;
    for (&d, &s) in dims.iter().zip(strides.iter()) {
        if d == 0 {
            return 0;
        }
        acc = acc.saturating_add((d - 1).saturating_mul(s));
    }
    acc
}

fn validate_within_capacity(
    base_offset: usize,
    dims: &[usize],
    strides: &[usize],
    capacity: usize,
) -> Result<(), ViewError> {
    // Maximum linear index accessed by the view is base_offset + max_index_offset
    let max_rel = max_index_offset(dims, strides);
    let required = base_offset.saturating_add(max_rel);
    if capacity == 0 {
        // Only valid if the view is also zero-sized (any dim 0)
        let zero_sized = dims.iter().copied().any(|d| d == 0) || dims.is_empty();
        if zero_sized {
            return Ok(());
        }
        return Err(ViewError::OutOfBounds {
            required_max_index: required,
            capacity,
        });
    }
    if required < capacity {
        Ok(())
    } else {
        Err(ViewError::OutOfBounds {
            required_max_index: required,
            capacity,
        })
    }
}

/// Create a reshape view over a contiguous tensor. Returns error if source is not contiguous
/// or if the total number of elements differs.
pub fn reshape_view(t: &Tensor, new_dims: &[usize]) -> Result<Tensor, ViewError> {
    let source_size = t.size();
    let requested: usize = new_dims.iter().product();
    let is_contig = t.is_contiguous();
    if !is_contig || requested != source_size {
        return Err(ViewError::InvalidReshape {
            requested: new_dims.to_vec(),
            source_size,
            is_contiguous: is_contig,
        });
    }

    // Reshape keeps same base pointer and owner; contiguous strides for new dims
    let shape = Shape::new(new_dims.to_vec());
    let device = t.device();
    let owner = t.allocation_owner().cloned();
    let base_ptr = unsafe { t.as_ptr() };
    Ok(Tensor::from_raw_view(base_ptr, shape, device, owner))
}

/// Create a transpose view by reordering dimensions and strides; no data movement.
pub fn transpose_view(t: &Tensor, perm: &[usize]) -> Result<Tensor, ViewError> {
    let rank = t.shape().rank();
    if perm.len() != rank {
        return Err(ViewError::InvalidPermutation {
            rank,
            perm: perm.to_vec(),
        });
    }
    // Validate it's a permutation of 0..rank
    {
        let mut seen = vec![false; rank];
        for &p in perm {
            if p >= rank || seen[p] {
                return Err(ViewError::InvalidPermutation {
                    rank,
                    perm: perm.to_vec(),
                });
            }
            seen[p] = true;
        }
    }

    let dims_src = t.shape().dims();
    let strides_src = t.shape().strides();
    let mut dims = Vec::with_capacity(rank);
    let mut strides = Vec::with_capacity(rank);
    for &p in perm {
        dims.push(dims_src[p]);
        strides.push(strides_src[p]);
    }

    // Capacity safety: transpose doesn't change base offset
    let capacity = capacity_elems_of(t);
    validate_within_capacity(0, &dims, &strides, capacity)?;

    let shape = Shape::with_strides(dims, strides);
    let device = t.device();
    let owner = t.allocation_owner().cloned();
    let base_ptr = unsafe { t.as_ptr() };
    Ok(Tensor::from_raw_view(base_ptr, shape, device, owner))
}

/// Create a 1D slice view over a contiguous base: start.. with step and length.
/// For multi-dimensional tensors, this treats the underlying memory as linear.
pub fn slice_view_linear(
    t: &Tensor,
    start: usize,
    step: usize,
    length: usize,
) -> Result<Tensor, ViewError> {
    if step == 0 {
        return Err(ViewError::InvalidSlice {
            start,
            step,
            length,
        });
    }
    if length == 0 {
        return reshape_view(t, &[0]);
    }

    // Base must be contiguous for simple linear slicing
    if !t.is_contiguous() {
        return Err(ViewError::InvalidSlice {
            start,
            step,
            length,
        });
    }

    let capacity = capacity_elems_of(t);
    let max_index = start.saturating_add((length - 1).saturating_mul(step));
    if capacity == 0 || max_index >= capacity {
        return Err(ViewError::OutOfBounds {
            required_max_index: max_index,
            capacity,
        });
    }

    // New dims and strides: [length], stride = step
    let dims = vec![length];
    let strides = vec![step];
    validate_within_capacity(start, &dims, &strides, capacity)?;

    let shape = Shape::as_view(dims, strides);
    let device = t.device();
    let owner = t.allocation_owner().cloned();
    let base_ptr = unsafe { t.as_ptr().add(start) };
    Ok(Tensor::from_raw_view(base_ptr, shape, device, owner))
}

/// General as_strided view; validates the region fits within allocation capacity.
pub fn as_strided_view(
    t: &Tensor,
    dims: &[usize],
    strides: &[usize],
    storage_offset: usize,
) -> Result<Tensor, ViewError> {
    if dims.len() != strides.len() {
        return Err(ViewError::InvalidReshape {
            requested: dims.to_vec(),
            source_size: t.size(),
            is_contiguous: t.is_contiguous(),
        });
    }
    let capacity = capacity_elems_of(t);
    validate_within_capacity(storage_offset, dims, strides, capacity)?;

    let shape = Shape::as_view(dims.to_vec(), strides.to_vec());
    let device = t.device();
    let owner = t.allocation_owner().cloned();
    let base_ptr = unsafe { t.as_ptr().add(storage_offset) };
    Ok(Tensor::from_raw_view(base_ptr, shape, device, owner))
}

/// Create an element view for the specified linear index.
pub fn element_view_linear(t: &Tensor, index: usize) -> Result<Tensor, ViewError> {
    // Iterate in logical view order, not raw memory order
    let size = t.size();
    if size == 0 || index >= size {
        return Err(ViewError::OutOfBounds {
            required_max_index: index,
            capacity: size,
        });
    }

    // Fast paths: contiguous or small fixed-rank shapes without heap allocation
    let rank = t.shape().rank();
    let rel_offset = if rank == 0 {
        0
    } else if t.is_contiguous() {
        // Contiguous linear view: direct index-to-offset mapping
        index
    } else {
        let dims = t.shape().dims();
        let strides = t.shape().strides();
        match rank {
            1 => {
                // Single dimension: offset = i * stride0
                index.saturating_mul(strides[0])
            }
            2 => {
                let d1 = dims[1];
                let i0 = index / d1;
                let i1 = index % d1;
                i0.saturating_mul(strides[0])
                    .saturating_add(i1.saturating_mul(strides[1]))
            }
            3 => {
                let d1 = dims[1];
                let d2 = dims[2];
                let plane = d1.saturating_mul(d2);
                let i0 = index / plane;
                let rem = index % plane;
                let i1 = rem / d2;
                let i2 = rem % d2;
                i0.saturating_mul(strides[0])
                    .saturating_add(i1.saturating_mul(strides[1]))
                    .saturating_add(i2.saturating_mul(strides[2]))
            }
            4 => {
                let d1 = dims[1];
                let d2 = dims[2];
                let d3 = dims[3];
                let block = d1.saturating_mul(d2).saturating_mul(d3);
                let i0 = index / block;
                let rem0 = index % block;
                let block2 = d2.saturating_mul(d3);
                let i1 = rem0 / block2;
                let rem1 = rem0 % block2;
                let i2 = rem1 / d3;
                let i3 = rem1 % d3;
                i0.saturating_mul(strides[0])
                    .saturating_add(i1.saturating_mul(strides[1]))
                    .saturating_add(i2.saturating_mul(strides[2]))
                    .saturating_add(i3.saturating_mul(strides[3]))
            }
            _ => {
                // General fallback: compute coordinates vector and use memory_offset
                let mut coords = vec![0usize; dims.len()];
                let mut lin = index;
                for d in (0..dims.len()).rev() {
                    let dim = dims[d];
                    coords[d] = if dim > 0 { lin % dim } else { 0 };
                    if dim > 0 {
                        lin /= dim;
                    }
                }
                t.memory_offset(&coords)
            }
        }
    };

    let shape = Shape::new(vec![1]);
    let device = t.device();
    let owner = t.allocation_owner().cloned();
    let base_ptr = unsafe { t.as_ptr().add(rel_offset) };
    Ok(Tensor::from_raw_view(base_ptr, shape, device, owner))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::core::memory::with_no_mem_padding;

    #[test]
    fn test_reshape_view_contiguous() {
        let t = Tensor::new(vec![4, 8]);
        let v = reshape_view(&t, &[2, 16]).unwrap();
        assert_eq!(v.shape().dims(), &[2, 16]);
        assert!(v.is_contiguous());
        assert_eq!(v.size(), t.size());
    }

    #[test]
    fn test_slice_view_linear_bounds_and_padding() {
        let base = Tensor::new(vec![100]);
        // Use a slice well within capacity
        let v = slice_view_linear(&base, 10, 2, 20).unwrap();
        assert_eq!(v.shape().dims(), &[20]);
        assert_eq!(v.shape().strides(), &[2]);

        // Out-of-bounds should error
        let cap = capacity_elems_of(&base);
        let too_far = cap + 1;
        let err = slice_view_linear(&base, too_far, 1, 1).unwrap_err();
        assert!(matches!(err, ViewError::OutOfBounds { .. }));
    }

    #[test]
    fn test_as_strided_and_transpose() {
        let t = Tensor::new(vec![4, 5]);
        // Transpose
        let tv = transpose_view(&t, &[1, 0]).unwrap();
        assert_eq!(tv.shape().dims(), &[5, 4]);
        assert_eq!(tv.shape().strides(), &[1, 5]);

        // As-strided view of a 2x3 window from linear memory
        let v = as_strided_view(&t, &[2, 3], &[5, 1], 0).unwrap();
        assert_eq!(v.shape().dims(), &[2, 3]);
        assert_eq!(v.shape().strides(), &[5, 1]);
    }

    #[test]
    fn test_element_view_linear_with_padding_modes() {
        // Padded (default)
        let t = Tensor::new(vec![33]);
        let v = element_view_linear(&t, 32).unwrap();
        assert_eq!(v.size(), 1);

        // No padding: bounds tie to logical size
        with_no_mem_padding(|| {
            let t = Tensor::new(vec![33]);
            // 33 is out of bounds without padding
            assert!(element_view_linear(&t, 33).is_err());
        });
    }

    #[test]
    fn test_gradtrack_through_views_and_ops() {
        let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();

        // element_view + add_scalar should register grad and propagate
        // Use high-level API to ensure grad registration
        let v = x.element_view(2);
        assert!(v.requires_grad());
        let mut y = v.add_scalar(5.0);
        y.backward(None);
        // Only index 2 receives gradient 1.0
        let gx = x.grad_owned().unwrap();
        assert_eq!(gx.data(), &[0.0, 0.0, 1.0, 0.0]);

        // Reshape via high-level API should register Reshape grad fn and backprop
        let x2 = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4])
            .unwrap()
            .with_requires_grad();
        let r = x2.view(vec![2, 2]);
        assert!(r.requires_grad());
        let mut s = r.sum();
        s.backward(None);
        let gx2 = x2.grad_owned().unwrap();
        assert_eq!(gx2.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_view_data_protection_on_source_inplace_modification() {
        let mut x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        // Create a slice view [2.0, 3.0]
        let v = slice_view_linear(&x, 1, 1, 2).unwrap();
        assert_eq!(v.data(), &[2.0, 3.0]);

        // Modify source in place
        {
            let d = x.data_mut();
            d[1] = 20.0;
            d[2] = 30.0;
        }
        // With copy-on-write, mutating the source does not affect existing views
        assert_eq!(v.data(), &[2.0, 3.0]);
    }
}
