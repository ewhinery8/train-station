//! Element iterator over tensor elements as zero-copy scalar views
//!
//! Gradients are preserved implicitly: each yielded scalar is created via
//! `Tensor::element_view`, which registers a `GradFn::View` with a
//! `ViewMapping::LinearRange { start, step: 1, length: 1 }`. When collecting
//! mapped elements back into a tensor and reducing (e.g., `sum()`), gradients
//! propagate correctly to the original tensor without extra flags.

use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

/// High-performance iterator over tensor elements as view tensors
///
/// Each element becomes a `Tensor` view of shape `[1]` that shares memory with
/// the source and preserves gradient tracking.
pub struct TensorElementIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) position: usize,
    pub(crate) end: usize,
}

impl<'a> TensorElementIterator<'a> {
    #[inline]
    pub fn new(tensor: &'a Tensor) -> Self {
        Self {
            source: tensor,
            position: 0,
            end: tensor.size(),
        }
    }

    #[inline]
    pub fn with_range(tensor: &'a Tensor, start: usize, end: usize) -> Self {
        let end = end.min(tensor.size());
        let start = start.min(end);
        Self {
            source: tensor,
            position: start,
            end,
        }
    }

    #[inline]
    fn create_element_view(&self, index: usize) -> Tensor {
        debug_assert!(index < self.source.size());
        self.source.element_view(index)
    }
}

impl<'a> Iterator for TensorElementIterator<'a> {
    type Item = Tensor;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position < self.end {
            let view = self.create_element_view(self.position);
            self.position += 1;
            Some(view)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.position;
        (remaining, Some(remaining))
    }

    #[inline]
    fn count(self) -> usize {
        self.end - self.position
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let new_pos = self.position.saturating_add(n);
        if new_pos < self.end {
            self.position = new_pos + 1;
            Some(self.create_element_view(new_pos))
        } else {
            self.position = self.end;
            None
        }
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        if self.position < self.end {
            let last_idx = self.end - 1;
            Some(self.create_element_view(last_idx))
        } else {
            None
        }
    }
}

impl<'a> ExactSizeIterator for TensorElementIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.position
    }
}

impl<'a> FusedIterator for TensorElementIterator<'a> {}

impl<'a> DoubleEndedIterator for TensorElementIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position < self.end {
            self.end -= 1;
            Some(self.create_element_view(self.end))
        } else {
            None
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let new_end = self.end.saturating_sub(n + 1);
        if new_end >= self.position {
            self.end = new_end;
            Some(self.create_element_view(self.end))
        } else {
            self.position = self.end;
            None
        }
    }
}

impl Tensor {
    /// Create an iterator over scalar elements (flattened view)
    ///
    /// Each yielded item is a `[1]`-shaped `Tensor` view that shares storage with
    /// the source. This iterator is GradTrack-aware; element operations propagate
    /// gradients to the original tensor when gradients are enabled.
    ///
    /// # Returns
    ///
    /// An iterator producing scalar view tensors in row-major order.
    ///
    /// # Examples
    ///
    /// Collect transformed elements back to the original shape using `collect_shape`:
    ///
    /// ```
    /// use train_station::tensor::TensorCollectExt;
    /// use train_station::Tensor;
    ///
    /// let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    /// let y = x
    ///     .iter_elements()
    ///     .map(|e| e.mul_scalar(2.0))
    ///     .collect_shape(vec![2, 2]);
    /// assert_eq!(y.data(), &[2.0, 4.0, 6.0, 8.0]);
    /// ```
    #[inline]
    pub fn iter_elements(&self) -> TensorElementIterator<'_> {
        TensorElementIterator::new(self)
    }

    /// Create an iterator over a clamped range of elements
    ///
    /// Produces scalar view tensors from `start..end` (clamped to `[0, size]`).
    ///
    /// # Arguments
    ///
    /// * `start` - Start index (inclusive)
    /// * `end` - End index (exclusive)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let x = Tensor::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), vec![6]).unwrap();
    /// let vals: Vec<f32> = x.iter_range(2, 5).map(|e| e.value()).collect();
    /// assert_eq!(vals, vec![2.0, 3.0, 4.0]);
    /// ```
    #[inline]
    pub fn iter_range(&self, start: usize, end: usize) -> TensorElementIterator<'_> {
        TensorElementIterator::with_range(self, start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradtrack::NoGradTrack;

    #[test]
    fn test_basic_iteration() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let vals: Vec<f32> = t.iter_elements().map(|e| e.value()).collect();
        assert_eq!(vals, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_range_iteration() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let vals: Vec<f32> = t.iter_range(1, 3).map(|e| e.value()).collect();
        assert_eq!(vals, vec![2.0, 3.0]);
    }

    #[test]
    fn test_iter_flat_gradient_propagation() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5])
            .unwrap()
            .with_requires_grad();

        // Map add_scalar over scalar views, collect, then sum
        let collected: Tensor = t.iter_elements().map(|e| e.add_scalar(1.0)).collect();
        let mut loss = collected.sum();
        loss.backward(None);

        let g = t.grad_owned().unwrap();
        assert_eq!(g.shape().dims(), vec![5]);
        assert_eq!(g.data(), &[1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_element_iterator_double_ended_and_exact_size() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let mut it = t.iter_elements();
        assert_eq!(it.len(), 4);
        assert_eq!(it.size_hint(), (4, Some(4)));
        assert_eq!(it.next_back().unwrap().value(), 4.0);
        assert_eq!(it.next().unwrap().value(), 1.0);
        assert_eq!(it.nth(1).unwrap().value(), 3.0);
        assert!(it.next().is_none());
        assert_eq!(t.iter_elements().last().unwrap().value(), 4.0);
    }

    #[test]
    fn test_iter_range_clamping_and_zero_sized() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3]).unwrap();
        let vals: Vec<f32> = t.iter_range(2, 10).map(|e| e.value()).collect();
        assert_eq!(vals, vec![3.0]);
        let empty = Tensor::new(vec![0]);
        let it = empty.iter_elements();
        assert_eq!(it.len(), 0);
        assert_eq!(it.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_iter_no_grad_guard() {
        let t = Tensor::from_slice(&[1.0, 2.0], vec![2])
            .unwrap()
            .with_requires_grad();
        let _g = NoGradTrack::new();
        let v = t.iter_elements().next().unwrap();
        assert!(!v.requires_grad());
    }
}
