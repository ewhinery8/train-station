//! Dimension iterator: iterate over sub-tensors along a specific dimension

// Grad tracking is registered by select; no direct use here
use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

/// Iterator that yields sub-tensors by slicing along a specific dimension.
pub struct TensorDimIterator<'a> {
    source: &'a Tensor,
    dim: usize,
    index: usize,
    end: usize,
}

impl<'a> TensorDimIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, dim: usize) -> Self {
        let rank = source.shape().rank();
        assert!(rank > 0, "iter_dim: cannot iterate dims of a 0-D tensor");
        let dim = if dim < rank {
            dim
        } else {
            panic!("iter_dim: dim {} out of bounds for rank {}", dim, rank)
        };
        let end = source.shape().dims()[dim];
        Self {
            source,
            dim,
            index: 0,
            end,
        }
    }

    #[inline]
    fn slice_at(&self, i: usize) -> Tensor {
        // Build view using select along dim i
        // Use existing select API if available; otherwise emulate via as_strided over remaining dims

        // Maintain gradient tracking - select already registers GradFn::Select
        self.source.select(self.dim, i)
    }
}

impl<'a> Iterator for TensorDimIterator<'a> {
    type Item = Tensor;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        let i = self.index;
        self.index += 1;
        Some(self.slice_at(i))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let rem = self.end - self.index;
        (rem, Some(rem))
    }
}

impl<'a> ExactSizeIterator for TensorDimIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.index
    }
}

impl<'a> FusedIterator for TensorDimIterator<'a> {}

impl<'a> std::iter::DoubleEndedIterator for TensorDimIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        self.end -= 1;
        Some(self.slice_at(self.end))
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        if self.index >= self.end {
            return None;
        }
        let new_end = self.end.saturating_sub(n + 1);
        if new_end < self.index {
            self.index = self.end;
            return None;
        }
        self.end = new_end;
        Some(self.slice_at(self.end))
    }
}

impl Tensor {
    /// Iterate over sub-tensors along a specific dimension.
    /// Each item is a view with that dimension removed (rank-1).
    #[inline]
    pub fn iter_dim(&self, dim: usize) -> TensorDimIterator<'_> {
        TensorDimIterator::new(self, dim)
    }
    /// Default iterator over the outermost dimension, yielding sub-tensors.
    #[inline]
    pub fn iter(&self) -> TensorDimIterator<'_> {
        self.iter_dim(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iter_dim_shapes() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let mut it0 = t.iter_dim(0);
        let a = it0.next().unwrap();
        let b = it0.next().unwrap();
        assert!(it0.next().is_none());
        assert_eq!(a.shape().dims(), vec![3]);
        assert_eq!(b.shape().dims(), vec![3]);
        assert_eq!(a.data(), &[1.0, 2.0, 3.0]);
        assert_eq!(b.data(), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_iter_dim_rank3() {
        let vals: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let t = Tensor::from_slice(&vals, vec![2, 3, 4]).unwrap();
        // iter over dim 1 → each item is [2,4]-shaped (with dim 1 removed)
        let v: Vec<Tensor> = t.iter_dim(1).collect();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].shape().rank(), 2);
        assert_eq!(v[0].shape().dims(), vec![2, 4]);
    }

    #[test]
    fn test_iter_dim_gradient_propagation() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2])
            .unwrap()
            .with_requires_grad();
        // operate per-row then collect and sum
        let collected: Tensor = t.iter_dim(0).map(|row| row.mul_scalar(2.0)).collect();
        let mut loss = collected.sum();
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        // Each element receives gradient 2.0 through the 2x scaling and sum
        assert_eq!(g.data(), &[2.0, 2.0, 2.0, 2.0]);
    }
}
