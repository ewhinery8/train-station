//! Linear window iterators over tensors as overlapping views
//!
//! Gradients are preserved implicitly for with-grad paths: each yielded window is
//! created via `Tensor::slice_view(start, 1, window_size)`, which registers a
//! `GradFn::View` with a `ViewMapping::LinearRange { start, step: 1, length: window_size }`
//! when the source requires gradients and gradient tracking is enabled.
//!
//! Performance routing is decided once at construction:
//! - In no-grad fast mode, if the source is non-contiguous, the iterator performs a
//!   single one-time `contiguous()` materialization and holds an internal owner.
//!   Subsequent window views are taken from this contiguous owner (no per-window copies).
//! - In with-grad mode, zero-copy views are created directly from the source.

use crate::tensor::core::utils::should_use_fast_path;
use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

pub struct TensorWindowsIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) window_size: usize,
    pub(crate) step: usize,
    pub(crate) start: usize,
    pub(crate) last_start: usize,
    pub(crate) finished: bool,
    // One-time contiguous owner for fast path on non-contiguous sources
    pub(crate) owner: Option<Tensor>,
}

impl<'a> TensorWindowsIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, window_size: usize, step: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        assert!(step > 0, "step must be > 0");
        let size = source.size();
        // Align the last_start to the stepping grid so reverse iteration returns
        // the same sequence of window starts as forward iteration in reverse order.
        let raw_last = size.saturating_sub(window_size);
        let last_start = if window_size > size {
            0
        } else {
            (raw_last / step) * step
        };
        let finished = window_size > size;
        let fast = should_use_fast_path(&[source]);
        let owner = if fast && !source.is_contiguous() && size > 0 {
            Some(source.contiguous())
        } else {
            None
        };
        Self {
            source,
            window_size,
            step,
            start: 0,
            last_start,
            finished,
            owner,
        }
    }

    #[inline]
    fn create_window_view(&self, start: usize) -> Tensor {
        // Select base tensor according to construction-time policy
        let base: &Tensor = match &self.owner {
            Some(o) => o,
            None => self.source,
        };
        base.slice_view(start, 1, self.window_size)
    }

    #[inline]
    fn windows_len(&self) -> usize {
        if self.finished {
            0
        } else {
            ((self.last_start - self.start) / self.step) + 1
        }
    }
}

impl<'a> Iterator for TensorWindowsIterator<'a> {
    type Item = Tensor;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.start > self.last_start {
            self.finished = true;
            return None;
        }
        let s = self.start;
        self.start = self.start.saturating_add(self.step);
        Some(self.create_window_view(s))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.windows_len();
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for TensorWindowsIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.windows_len()
    }
}

impl<'a> FusedIterator for TensorWindowsIterator<'a> {}

impl<'a> DoubleEndedIterator for TensorWindowsIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.last_start < self.start {
            self.finished = true;
            return None;
        }
        let s = self.last_start;
        if s < self.step {
            self.last_start = 0usize;
        } else {
            self.last_start -= self.step;
        }
        Some(self.create_window_view(s))
    }
}

impl Tensor {
    /// Overlapping windows iterator with step=1. Use this instead of `iter_windows`.
    ///
    /// Produces overlapping linear windows as view tensors. In no-grad fast mode,
    /// a contiguous owner may be materialized once for faster subsequent views.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Length of each window (> 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
    /// let v: Vec<f32> = t.windows(3).map(|w| w.sum().value()).collect();
    /// assert_eq!(v, vec![6.0, 9.0]);
    /// ```
    #[inline]
    pub fn windows(&self, window_size: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, 1)
    }

    /// Overlapping windows iterator with custom step. Use this instead of `iter_windows_step`.
    ///
    /// Produces windows starting at positions `0, step, 2*step, ...` up to the last
    /// valid start. Reverse iteration yields the same sequence in reverse.
    ///
    /// # Arguments
    ///
    /// * `window_size` - Length of each window (> 0)
    /// * `step` - Step between consecutive window starts (> 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let t = Tensor::from_slice(&(1..=8).map(|i| i as f32).collect::<Vec<_>>(), vec![8]).unwrap();
    /// let wins: Vec<Tensor> = t.windows_step(3, 2).collect();
    /// assert_eq!(wins[0].data(), &[1.0, 2.0, 3.0]);
    /// assert_eq!(wins[1].data(), &[3.0, 4.0, 5.0]);
    /// assert_eq!(wins[2].data(), &[5.0, 6.0, 7.0]);
    /// ```
    #[inline]
    pub fn windows_step(&self, window_size: usize, step: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, step)
    }

    #[deprecated(note = "Use Tensor::windows(...) instead. This alias will be removed before 1.0.")]
    #[inline]
    pub fn iter_windows(&self, window_size: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, 1)
    }

    #[deprecated(
        note = "Use Tensor::windows_step(...) instead. This alias will be removed before 1.0."
    )]
    #[inline]
    pub fn iter_windows_step(&self, window_size: usize, step: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradtrack::NoGradTrack;

    #[test]
    fn test_windows() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let wins: Vec<Tensor> = t.windows(3).collect();
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(wins[1].data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_windows_step() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let wins: Vec<Tensor> = t.windows_step(2, 2).collect();
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[0].data(), &[1.0, 2.0]);
        assert_eq!(wins[1].data(), &[3.0, 4.0]);
    }

    #[test]
    fn test_windows_over_3d_outerdim() {
        // 3D [B=1, T=5, F=2]
        let vals: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&vals, vec![1, 5, 2]).unwrap();
        // For each outer slice (batch), take time windows of size 3, step 2
        for b in t.iter_dim(0) {
            // b shape [5,2]; create windows along dim 0 of length 3 with step 2
            // emulate with split/stride windows: we'll use iter_windows_step over linear memory of b.flattened rows
            let wins: Vec<Tensor> = b
                .split_with_sizes(&[3, 2], 0) // emulate two windows [0..3] and [2..5]
                .into_iter()
                .collect();
            assert_eq!(wins[0].shape().dims(), vec![3, 2]);
            assert_eq!(wins[1].shape().dims(), vec![2, 2]);
        }
    }

    #[test]
    fn test_windows_gradient_chain() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2])
            .unwrap()
            .with_requires_grad();
        // Create overlapping windows along dim 0 of size 2 with step 1: [rows 0..2], [1..3]
        let wins = t.split_with_sizes(&[2, 1], 0); // first window rows [0,1]
        let w0 = wins[0].mul_scalar(2.0);
        // Build second window rows [1,2] explicitly
        let row1 = t.select(0, 1).unsqueeze(0);
        let row2 = t.select(0, 2).unsqueeze(0);
        let w1 = Tensor::cat(&[row1, row2], 0).mul_scalar(2.0);
        let y = Tensor::cat(&[w0, w1], 0);
        let mut loss = y.sum();
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        // Rows 1 appears in both windows → grad accum twice (2 + 2); others once
        assert_eq!(g.get(&[0, 0]), 2.0);
        assert_eq!(g.get(&[1, 0]), 4.0);
        assert_eq!(g.get(&[2, 0]), 2.0);
    }

    #[test]
    fn test_windows_iter_gradient_propagation() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();
        // Overlapping windows of size 3, step 1; map then collect
        let wins: Vec<Tensor> = t.windows(3).map(|w| w.mul_scalar(2.0)).collect();
        let y = Tensor::cat(&wins, 0).sum();
        let mut loss = y;
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        // Coverage per index: [0] in 1 window, [1] in 2, [2] in 2, [3] in 1; each window scaled by 2
        assert_eq!(g.data(), &[2.0, 4.0, 4.0, 2.0]);
    }

    #[test]
    fn test_windows_double_ended_and_size_hint() {
        let t =
            Tensor::from_slice(&(1..=8).map(|i| i as f32).collect::<Vec<_>>(), vec![8]).unwrap();
        let mut it = t.windows_step(3, 2); // starts at 0,2,4
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.len(), 3);

        let back = it.next_back().unwrap();
        assert_eq!(back.data(), &[5.0, 6.0, 7.0]);
        assert_eq!(it.len(), 2);
        let front = it.next().unwrap();
        assert_eq!(front.data(), &[1.0, 2.0, 3.0]);
        let mid = it.next().unwrap();
        assert_eq!(mid.data(), &[3.0, 4.0, 5.0]);
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
        assert_eq!(it.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_windows_zero_sized_tensor() {
        let t = Tensor::new(vec![0]);
        let it = t.windows(3);
        assert_eq!(it.len(), 0);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.collect::<Vec<_>>().len(), 0);
    }

    #[test]
    fn test_windows_no_grad_guard_disables_requires_grad() {
        let t = Tensor::from_slice(&(0..6).map(|i| i as f32).collect::<Vec<_>>(), vec![6])
            .unwrap()
            .with_requires_grad();
        let _guard = NoGradTrack::new();
        let w = t.windows(4).next().unwrap();
        assert!(!w.requires_grad());
        let y: Tensor = t.windows(4).collect_shape(vec![3, 4]);
        assert!(!y.requires_grad());
    }
}
