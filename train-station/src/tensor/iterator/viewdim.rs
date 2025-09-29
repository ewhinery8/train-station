//! Dimension iterator: iterate over sub-tensors along a specific dimension

// Grad tracking is registered by select; no direct use here
use crate::tensor::core::utils::should_use_fast_path;
use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

/// Iterator that yields sub-tensors by slicing along a specific dimension.
pub struct TensorDimIterator<'a> {
    source: &'a Tensor,
    dim: usize,
    index: usize,
    end: usize,
    // One-time contiguous owner for fast path on non-contiguous sources
    owner: Option<Tensor>,
    // For 1D tensors, prefer element views for better grad semantics and collection perf
    use_element_views: bool,
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
        // Decide fast path once at construction (no-grad or inference)
        let fast = should_use_fast_path(&[source]);
        // One-time contiguous materialization for better cache/linearization
        let owner = if fast && !source.is_contiguous() && source.size() > 0 {
            Some(source.contiguous())
        } else {
            None
        };
        Self {
            source,
            dim,
            index: 0,
            end,
            owner,
            use_element_views: rank == 1,
        }
    }

    #[inline]
    fn slice_at(&self, i: usize) -> Tensor {
        // Choose base tensor once according to construction-time policy
        let base: &Tensor = match &self.owner {
            Some(o) => o,
            None => self.source,
        };
        if self.use_element_views {
            // 1D optimization: scalar element views cooperate with grad tracking
            base.element_view(i)
        } else {
            // Multi-dim: rely on select for shape/stride-correct subviews
            base.select(self.dim, i)
        }
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
    ///
    /// Produces view tensors by slicing along the given dimension; each item
    /// has that dimension removed (rank - 1). Views share storage and preserve
    /// gradient tracking semantics.
    ///
    /// # Arguments
    ///
    /// * `dim` - Dimension to iterate over
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::TensorCollectExt;
    /// use train_station::Tensor;
    ///
    /// let t = Tensor::from_slice(&(1..=6).map(|i| i as f32).collect::<Vec<_>>(), vec![2, 3]).unwrap();
    /// let out = t.iter_dim(0).map(|row| row.add_scalar(1.0)).collect_shape(vec![2, 3]);
    /// assert_eq!(out.data(), &[2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    /// ```
    #[inline]
    pub fn iter_dim(&self, dim: usize) -> TensorDimIterator<'_> {
        TensorDimIterator::new(self, dim)
    }

    /// Default iterator over the outermost dimension, yielding sub-tensors (N-D) or scalar views (1-D).
    ///
    /// This is equivalent to `iter_dim(0)` with an important optimization:
    /// - For 1-D tensors, it yields scalar element views of shape `[1]` (same as `iter_flat()`)
    ///   to maximize GradTrack cooperation and collection performance.
    /// - For N-D tensors (rank > 1), it yields sub-tensors with the outermost dimension removed
    ///   (rank − 1), suitable for row/batch-wise processing.
    ///
    /// Views share storage with the source tensor and preserve gradient tracking semantics.
    /// Use `collect_shape([..])` to reconstruct shape efficiently after per-item transforms.
    ///
    /// # Returns
    ///
    /// An iterator producing view tensors for each slice along the outermost dimension
    /// (or scalar views for 1-D).
    ///
    /// # Examples
    ///
    /// 1-D: element views (shape `[1]`) and shape-preserving collection
    ///
    /// ```
    /// use train_station::tensor::TensorCollectExt;
    /// use train_station::Tensor;
    ///
    /// let v = Tensor::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), vec![6]).unwrap();
    /// let out = v.iter().map(|e| e.add_scalar(1.0)).collect_shape(vec![6]);
    /// assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    /// ```
    ///
    /// 2-D: row-wise transforms and shape-preserving collection
    ///
    /// ```
    /// use train_station::tensor::TensorCollectExt;
    /// use train_station::Tensor;
    ///
    /// let m = Tensor::from_slice(&(1..=6).map(|i| i as f32).collect::<Vec<_>>(), vec![2, 3]).unwrap();
    /// let y = m.iter().map(|row| row.mul_scalar(2.0)).collect_shape(vec![2, 3]);
    /// assert_eq!(y.data(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    #[inline]
    pub fn iter(&self) -> TensorDimIterator<'_> {
        self.iter_dim(0)
    }

    /// Explicit alias for outermost-dimension iteration of sub-tensors.
    /// Equivalent to `iter_dim(0)`.
    #[inline]
    pub fn outer_iter(&self) -> TensorDimIterator<'_> {
        self.iter_dim(0)
    }
}

/// Owned variant of TensorDimIterator enabling `IntoIterator for Tensor` and iterator flattening
pub struct TensorDimOwnedIterator {
    owner: Tensor,
    dim: usize,
    index: usize,
    end: usize,
    // For 1D tensors, using element views cooperates best with grad tracking and collection perf
    use_element_views: bool,
}

impl TensorDimOwnedIterator {
    #[inline]
    pub fn new(source: Tensor, dim: usize) -> Self {
        let rank = source.shape().rank();
        assert!(rank > 0, "iter_dim: cannot iterate dims of a 0-D tensor");
        let dim = if dim < rank {
            dim
        } else {
            panic!("iter_dim: dim {} out of bounds for rank {}", dim, rank)
        };
        let end = source.shape().dims()[dim];
        let fast = should_use_fast_path(&[&source]);
        let owner = if fast && !source.is_contiguous() && source.size() > 0 {
            source.contiguous()
        } else {
            source
        };
        Self {
            owner,
            dim,
            index: 0,
            end,
            use_element_views: rank == 1,
        }
    }

    #[inline]
    fn slice_at(&self, i: usize) -> Tensor {
        if self.use_element_views {
            self.owner.element_view(i)
        } else {
            self.owner.select(self.dim, i)
        }
    }
}

impl Iterator for TensorDimOwnedIterator {
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

impl ExactSizeIterator for TensorDimOwnedIterator {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.index
    }
}

impl FusedIterator for TensorDimOwnedIterator {}

impl std::iter::DoubleEndedIterator for TensorDimOwnedIterator {
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
