//! Linear chunk iterators over tensors as contiguous views

use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

pub struct TensorChunksIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) chunk_size: usize,
    pub(crate) position: usize,
    pub(crate) end: usize,
}

impl<'a> TensorChunksIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        Self {
            source,
            chunk_size,
            position: 0,
            end: source.size(),
        }
    }

    #[inline]
    fn create_chunk_view(&self, start: usize, len: usize) -> Tensor {
        if len == 0 {
            return Tensor::new(vec![0]);
        }
        // Force contiguous slice to avoid stepped views exposing stride>1 in iterators
        let v = self.source.slice_view(start, 1, len);
        if v.is_contiguous() {
            v
        } else {
            v.contiguous()
        }
    }
}

impl<'a> Iterator for TensorChunksIterator<'a> {
    type Item = Tensor;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.end {
            return None;
        }
        let start = self.position;
        let remaining = self.end - start;
        let take = remaining.min(self.chunk_size);
        self.position += take;
        Some(self.create_chunk_view(start, take))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end.saturating_sub(self.position);
        let n = if remaining == 0 {
            0
        } else {
            remaining.div_ceil(self.chunk_size)
        };
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for TensorChunksIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        let remaining = self.end.saturating_sub(self.position);
        if remaining == 0 {
            0
        } else {
            remaining.div_ceil(self.chunk_size)
        }
    }
}

impl<'a> FusedIterator for TensorChunksIterator<'a> {}

impl<'a> DoubleEndedIterator for TensorChunksIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position >= self.end {
            return None;
        }
        let remaining = self.end - self.position;
        let take = remaining.min(self.chunk_size);
        self.end -= take;
        Some(self.create_chunk_view(self.end, take))
    }
}

pub struct TensorChunksExactIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) chunk_size: usize,
    pub(crate) position: usize,
    pub(crate) exact_end: usize,
    pub(crate) remainder_start: usize,
    pub(crate) remainder_len: usize,
}

impl<'a> TensorChunksExactIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        let size = source.size();
        let exact_chunks = size / chunk_size;
        let exact_end = exact_chunks * chunk_size;
        let remainder_len = size - exact_end;
        Self {
            source,
            chunk_size,
            position: 0,
            exact_end,
            remainder_start: exact_end,
            remainder_len,
        }
    }

    #[inline]
    pub fn remainder(&self) -> Tensor {
        if self.remainder_len == 0 {
            Tensor::new(vec![0])
        } else {
            let v = self
                .source
                .slice_view(self.remainder_start, 1, self.remainder_len);
            if v.is_contiguous() {
                v
            } else {
                v.contiguous()
            }
        }
    }

    #[inline]
    fn create_chunk_view(&self, start: usize) -> Tensor {
        let v = self.source.slice_view(start, 1, self.chunk_size);
        if v.is_contiguous() {
            v
        } else {
            v.contiguous()
        }
    }
}

impl<'a> Iterator for TensorChunksExactIterator<'a> {
    type Item = Tensor;
    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.position >= self.exact_end {
            return None;
        }
        let start = self.position;
        self.position += self.chunk_size;
        Some(self.create_chunk_view(start))
    }
    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.exact_end.saturating_sub(self.position);
        let n = remaining / self.chunk_size;
        (n, Some(n))
    }
}

impl<'a> ExactSizeIterator for TensorChunksExactIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        (self.exact_end.saturating_sub(self.position)) / self.chunk_size
    }
}

impl<'a> FusedIterator for TensorChunksExactIterator<'a> {}

impl<'a> DoubleEndedIterator for TensorChunksExactIterator<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.position >= self.exact_end {
            return None;
        }
        self.exact_end = self.exact_end.saturating_sub(self.chunk_size);
        Some(self.create_chunk_view(self.exact_end))
    }
}

impl Tensor {
    #[inline]
    pub fn iter_chunks(&self, chunk_size: usize) -> TensorChunksIterator<'_> {
        TensorChunksIterator::new(self, chunk_size)
    }

    #[inline]
    pub fn iter_chunks_exact(&self, chunk_size: usize) -> TensorChunksExactIterator<'_> {
        TensorChunksExactIterator::new(self, chunk_size)
    }

    /// Iterate with an auto-tuned chunk size for cache-friendly processing
    ///
    /// Heuristic:
    /// - Target ~64 KiB blocks (16K f32 elements) for good L1/L2 behavior
    /// - Clamp to [4K, 262_144] elements
    /// - Round to a multiple of SIMD lane width when possible
    #[inline]
    pub fn iter_fast_chunks(&self) -> TensorChunksIterator<'_> {
        let n = self.size();
        if n == 0 {
            return TensorChunksIterator::new(self, 1);
        }
        // 64 KiB in f32 elements
        let mut sz = 16_384usize;
        // Adjust by tensor size rough scale
        if n < 16_384 {
            sz = 4_096;
        }
        if n > 1_048_576 {
            sz = 65_536;
        }
        // Align to SIMD lane if available
        let lane = crate::tensor::core::Tensor::simd_lane_width_elems_runtime();
        if lane > 1 {
            sz = sz.div_ceil(lane) * lane;
        }
        // Clamp
        sz = sz.clamp(4_096, 262_144);
        TensorChunksIterator::new(self, sz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunks() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let v: Vec<Tensor> = t.iter_chunks(2).collect();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].data(), &[1.0, 2.0]);
        assert_eq!(v[1].data(), &[3.0, 4.0]);
        assert_eq!(v[2].data(), &[5.0]);
    }

    #[test]
    fn test_chunks_exact() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let mut it = t.iter_chunks_exact(2);
        let a = it.next().unwrap();
        let b = it.next().unwrap();
        assert!(it.next().is_none());
        assert_eq!(a.data(), &[10.0, 20.0]);
        assert_eq!(b.data(), &[30.0, 40.0]);
        assert_eq!(it.remainder().data(), &[50.0]);
    }

    #[test]
    fn test_chunks_over_3d_outerdim() {
        // 3D tensor [B=2, T=3, F=4]
        let vals: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let t = Tensor::from_slice(&vals, vec![2, 3, 4]).unwrap();
        // Iterate outermost dim via iter_dim(0), then chunk time dimension T with size 2
        let mut collected: Vec<Vec<Tensor>> = Vec::new();
        for b in t.iter_dim(0) {
            let chunks: Vec<Tensor> = b.split(2, 0); // split along current outer dim (was T)
                                                     // Each chunk should have shape [2,4] then [1,4]
            assert_eq!(chunks[0].shape().dims(), vec![2, 4]);
            assert_eq!(chunks[1].shape().dims(), vec![1, 4]);
            collected.push(chunks);
        }
        assert_eq!(collected.len(), 2);
    }

    #[test]
    fn test_chunks_gradient_after_collect() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3])
            .unwrap()
            .with_requires_grad();
        // Split columns in chunks of 2, operate per chunk, re-concat and sum
        let parts = t.split(2, 1);
        let parts2: Vec<Tensor> = parts.into_iter().map(|p| p.mul_scalar(3.0)).collect();
        let y = Tensor::cat(&parts2, 1);
        let mut loss = y.sum();
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        // Each element gets grad 3.0
        assert_eq!(g.data(), &[3.0, 3.0, 3.0, 3.0, 3.0, 3.0]);
    }
}
