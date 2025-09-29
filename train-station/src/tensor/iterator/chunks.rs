//! Linear chunk iterators over tensors as contiguous views
//!
//! Gradients are preserved implicitly for with-grad paths: each yielded chunk is
//! created via `Tensor::slice_view(start, 1, len)`, which registers a
//! `GradFn::View` with a `ViewMapping::LinearRange { start, step: 1, length: len }`
//! when the source requires gradients and gradient tracking is enabled.
//!
//! Performance routing is decided once at construction:
//! - In no-grad fast mode, if the source is non-contiguous, the iterator performs a
//!   single one-time `contiguous()` materialization and holds an internal owner.
//!   Subsequent chunk views are taken from this contiguous owner (no per-chunk copies).
//! - In with-grad mode, zero-copy views are created directly from the source.

use crate::tensor::core::utils::should_use_fast_path;
use crate::tensor::core::Tensor;
use std::iter::{ExactSizeIterator, FusedIterator};

pub struct TensorChunksIterator<'a> {
    pub(crate) source: &'a Tensor,
    pub(crate) chunk_size: usize,
    pub(crate) position: usize,
    pub(crate) end: usize,
    // One-time contiguous owner for fast path on non-contiguous sources
    pub(crate) owner: Option<Tensor>,
}

impl<'a> TensorChunksIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        let fast = should_use_fast_path(&[source]);
        // One-time contiguous materialization policy for fast path
        let owner = if fast && !source.is_contiguous() && source.size() > 0 {
            Some(source.contiguous())
        } else {
            None
        };
        Self {
            source,
            chunk_size,
            position: 0,
            end: source.size(),
            owner,
        }
    }

    #[inline]
    fn create_chunk_view(&self, start: usize, len: usize) -> Tensor {
        if len == 0 {
            return Tensor::new(vec![0]);
        }
        // Choose base tensor once according to construction-time policy
        let base: &Tensor = match &self.owner {
            Some(o) => o,
            None => self.source,
        };
        base.slice_view(start, 1, len)
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
        // Match standard slice chunk semantics: the last chunk may be a smaller remainder
        let rem = remaining % self.chunk_size;
        let take = if rem == 0 { self.chunk_size } else { rem };
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
    // One-time contiguous owner for fast path on non-contiguous sources
    pub(crate) owner: Option<Tensor>,
}

impl<'a> TensorChunksExactIterator<'a> {
    #[inline]
    pub fn new(source: &'a Tensor, chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "chunk_size must be > 0");
        let size = source.size();
        let exact_chunks = size / chunk_size;
        let exact_end = exact_chunks * chunk_size;
        let remainder_len = size - exact_end;
        let fast = should_use_fast_path(&[source]);
        let owner = if fast && !source.is_contiguous() && size > 0 {
            Some(source.contiguous())
        } else {
            None
        };
        Self {
            source,
            chunk_size,
            position: 0,
            exact_end,
            remainder_start: exact_end,
            remainder_len,
            owner,
        }
    }

    #[inline]
    pub fn remainder(&self) -> Tensor {
        if self.remainder_len == 0 {
            Tensor::new(vec![0])
        } else {
            let base: &Tensor = match &self.owner {
                Some(o) => o,
                None => self.source,
            };
            base.slice_view(self.remainder_start, 1, self.remainder_len)
        }
    }

    #[inline]
    fn create_chunk_view(&self, start: usize) -> Tensor {
        let base: &Tensor = match &self.owner {
            Some(o) => o,
            None => self.source,
        };
        base.slice_view(start, 1, self.chunk_size)
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
    /// Standard slice-like chunks iterator. Use this instead of `iter_chunks`.
    ///
    /// Iterates over contiguous or view-backed slices of the tensor with the
    /// specified chunk size. In no-grad fast mode, a single contiguous owner may
    /// be materialized to optimize subsequent views.
    ///
    /// # Arguments
    ///
    /// * `chunk_size` - Number of elements per chunk (must be > 0)
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::tensor::TensorCollectExt;
    /// use train_station::Tensor;
    ///
    /// let t = Tensor::from_slice(&(1..=6).map(|i| i as f32).collect::<Vec<_>>(), vec![6]).unwrap();
    /// let y = t.chunks(2).map(|c| c.mul_scalar(2.0)).collect_shape(vec![6]);
    /// assert_eq!(y.data(), &[2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    /// ```
    #[inline]
    pub fn chunks(&self, chunk_size: usize) -> TensorChunksIterator<'_> {
        TensorChunksIterator::new(self, chunk_size)
    }

    /// Standard slice-like exact chunks iterator. Use this instead of `iter_chunks_exact`.
    ///
    /// Yields only the exact chunks of size `chunk_size`, exposing any remainder
    /// via `remainder()`. See `chunks()` for a variant that yields the remainder as
    /// the last (smaller) chunk.
    ///
    /// # Examples
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
    /// let mut it = t.chunks_exact(2);
    /// assert_eq!(it.next().unwrap().data(), &[1.0, 2.0]);
    /// assert_eq!(it.next().unwrap().data(), &[3.0, 4.0]);
    /// assert_eq!(it.remainder().data(), &[5.0]);
    /// ```
    #[inline]
    pub fn chunks_exact(&self, chunk_size: usize) -> TensorChunksExactIterator<'_> {
        TensorChunksExactIterator::new(self, chunk_size)
    }

    #[deprecated(note = "Use Tensor::chunks(...) instead. This alias will be removed before 1.0.")]
    #[inline]
    pub fn iter_chunks(&self, chunk_size: usize) -> TensorChunksIterator<'_> {
        TensorChunksIterator::new(self, chunk_size)
    }

    #[deprecated(
        note = "Use Tensor::chunks_exact(...) instead. This alias will be removed before 1.0."
    )]
    #[inline]
    pub fn iter_chunks_exact(&self, chunk_size: usize) -> TensorChunksExactIterator<'_> {
        TensorChunksExactIterator::new(self, chunk_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradtrack::NoGradTrack;

    #[test]
    fn test_chunks() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let v: Vec<Tensor> = t.chunks(2).collect();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].data(), &[1.0, 2.0]);
        assert_eq!(v[1].data(), &[3.0, 4.0]);
        assert_eq!(v[2].data(), &[5.0]);
    }

    #[test]
    fn test_chunks_exact() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0, 50.0], vec![5]).unwrap();
        let mut it = t.chunks_exact(2);
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

    #[test]
    fn test_chunks_iter_gradient_propagation() {
        let t = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![4])
            .unwrap()
            .with_requires_grad();
        // Use chunks iterator over 2-sized windows, map op, collect and sum
        let parts: Vec<Tensor> = t.chunks(2).map(|c| c.add_scalar(1.0)).collect();
        let y = Tensor::cat(&parts, 0);
        let mut loss = y.sum();
        loss.backward(None);
        let g = t.grad_owned().unwrap();
        assert_eq!(g.data(), &[1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_chunks_double_ended_and_size_hint() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0], vec![5]).unwrap();
        let mut it = t.chunks(2);
        assert_eq!(it.size_hint(), (3, Some(3)));
        assert_eq!(it.len(), 3);

        let last = it.next_back().unwrap();
        assert_eq!(last.data(), &[5.0]);
        assert_eq!(it.len(), 2);

        let first = it.next().unwrap();
        assert_eq!(first.data(), &[1.0, 2.0]);
        assert_eq!(it.len(), 1);

        let middle = it.next().unwrap();
        assert_eq!(middle.data(), &[3.0, 4.0]);
        assert!(it.next().is_none());
        assert!(it.next_back().is_none());
        assert_eq!(it.size_hint(), (0, Some(0)));
    }

    #[test]
    fn test_chunks_zero_sized_tensor() {
        let t = Tensor::new(vec![0]);
        let it = t.chunks(3);
        assert_eq!(it.len(), 0);
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert_eq!(it.collect::<Vec<_>>().len(), 0);
    }

    #[test]
    fn test_chunks_no_grad_guard_disables_requires_grad() {
        let t = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4])
            .unwrap()
            .with_requires_grad();
        let _guard = NoGradTrack::new();
        let mut it = t.chunks(2);
        let a = it.next().unwrap();
        let b = it.next().unwrap();
        assert!(!a.requires_grad());
        assert!(!b.requires_grad());
        // Collect under guard should produce no-grad result
        let y: Tensor = t.chunks(2).collect_shape(vec![2, 2]);
        assert!(!y.requires_grad());
    }
}
