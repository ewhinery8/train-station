//! Element iterator over tensor elements as zero-copy scalar views

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
    #[inline]
    pub fn iter_elements(&self) -> TensorElementIterator<'_> {
        TensorElementIterator::new(self)
    }

    /// Create an iterator over a clamped range of elements
    #[inline]
    pub fn iter_range(&self, start: usize, end: usize) -> TensorElementIterator<'_> {
        TensorElementIterator::with_range(self, start, end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
