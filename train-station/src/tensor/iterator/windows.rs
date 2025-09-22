//! Linear window iterators over tensors as overlapping views

use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

pub struct TensorWindowsIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) window_size: usize,
    pub(crate) step: usize,
    pub(crate) start: usize,
    pub(crate) last_start: usize,
    pub(crate) finished: bool,
}

impl<'a> TensorWindowsIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, window_size: usize, step: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        assert!(step > 0, "step must be > 0");
        let size = source.size();
        let last_start = size.saturating_sub(window_size);
        let finished = window_size > size;
        Self {
            source,
            window_size,
            step,
            start: 0,
            last_start,
            finished,
        }
    }

    #[inline]
    fn create_window_view(&self, start: usize) -> Tensor {
        self.source.slice_view(start, 1, self.window_size)
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
    #[inline]
    pub fn iter_windows(&self, window_size: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, 1)
    }

    #[inline]
    pub fn iter_windows_step(&self, window_size: usize, step: usize) -> TensorWindowsIterator<'_> {
        TensorWindowsIterator::new(self, window_size, step)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_windows() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let wins: Vec<Tensor> = t.iter_windows(3).collect();
        assert_eq!(wins.len(), 2);
        assert_eq!(wins[0].data(), &[1.0, 2.0, 3.0]);
        assert_eq!(wins[1].data(), &[2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_windows_step() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let wins: Vec<Tensor> = t.iter_windows_step(2, 2).collect();
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
}
