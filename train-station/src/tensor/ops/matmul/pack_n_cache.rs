//! Panel packing and caching architecture for high-performance matrix multiplication
//!
//! This module implements cache-friendly panel packing strategies based on BLIS/GOTOBLAS
//! algorithms. It provides optimized memory layouts for A and B matrix panels with
//! intelligent caching and reuse for batched operations.

use crate::tensor::core::memory::{detect_runtime_simd, SimdLevel};
use crate::tensor::Tensor;
use std::sync::OnceLock;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::__cpuid_count;

/// Cache blocking parameters optimized for different cache levels
#[derive(Debug, Clone, Copy)]
pub struct BlockingParams {
    /// L3 cache blocking for N dimension (typically full N)
    #[allow(dead_code)]
    pub nc: usize,
    /// L2 cache blocking for K dimension (240 for AVX2, 480 for AVX512)
    pub kc: usize,
    /// L2 cache blocking for M dimension (120 for AVX2, 240 for AVX512)
    pub mc: usize,
    /// Register blocking for N dimension (16 for AVX2, 32 for AVX512)
    pub nr: usize,
    /// Register blocking for M dimension (6 for AVX2, 8 for AVX512)
    pub mr: usize,
}

impl BlockingParams {
    /// Get optimal blocking parameters for current SIMD level
    pub fn for_simd_level(level: SimdLevel) -> Self {
        // Try CPUID-based autotuning first; fall back to good defaults per SIMD level
        if let Some(caches) = detect_cpu_caches() {
            return tuned_for(level, caches);
        }

        match level {
            SimdLevel::Avx512 => tuned_for(
                level,
                CpuCaches {
                    l1d_bytes: 32 * 1024,
                    l2_bytes: 512 * 1024,
                    l3_bytes: 8 * 1024 * 1024,
                },
            ),
            SimdLevel::Avx2 => tuned_for(
                level,
                CpuCaches {
                    l1d_bytes: 32 * 1024,
                    l2_bytes: 256 * 1024,
                    l3_bytes: 8 * 1024 * 1024,
                },
            ),
            SimdLevel::Sse2 => tuned_for(
                level,
                CpuCaches {
                    l1d_bytes: 32 * 1024,
                    l2_bytes: 256 * 1024,
                    l3_bytes: 4 * 1024 * 1024,
                },
            ),
            SimdLevel::Scalar => tuned_for(
                level,
                CpuCaches {
                    l1d_bytes: 16 * 1024,
                    l2_bytes: 128 * 1024,
                    l3_bytes: 2 * 1024 * 1024,
                },
            ),
        }
    }

    /// Get cached blocking parameters for current runtime SIMD level
    pub fn get_cached() -> &'static BlockingParams {
        static CACHED_PARAMS: OnceLock<BlockingParams> = OnceLock::new();
        CACHED_PARAMS.get_or_init(|| Self::for_simd_level(detect_runtime_simd()))
    }
}

#[derive(Debug, Clone, Copy)]
struct CpuCaches {
    l1d_bytes: usize,
    l2_bytes: usize,
    l3_bytes: usize,
}

fn tuned_for(level: SimdLevel, caches: CpuCaches) -> BlockingParams {
    // Helper to round up/down to multiples
    fn floor_to_multiple(x: usize, mult: usize) -> usize {
        if mult == 0 {
            return x;
        }
        x / mult * mult
    }

    match level {
        SimdLevel::Avx512 => {
            // AVX512: 16-wide nr typical
            let nr = 32usize;
            let mr = 8usize;
            // kc so that kc*nr*4 fits comfortably in L1D (leave 25% headroom)
            let kc_from_l1 = ((caches.l1d_bytes * 3 / 4) / (nr * 4)).max(64);
            let kc = floor_to_multiple(kc_from_l1.min(1024), 16).max(64);
            // mc so that kc*mc*4 <= ~3/4 L2
            let mc_from_l2 = ((caches.l2_bytes * 3 / 4) / (kc * 4)).max(mr);
            let mc = floor_to_multiple(mc_from_l2.min(512), mr).max(mr);
            // nc so that kc*nc*4 <= ~7/8 L3 (or reasonable default if no L3)
            let l3 = caches.l3_bytes.max(1);
            let nc_from_l3 = ((l3 * 7 / 8) / (kc * 4)).max(nr * 8);
            let nc = floor_to_multiple(nc_from_l3.min(4096), nr).max(nr * 4);

            BlockingParams { nc, kc, mc, nr, mr }
        }
        SimdLevel::Avx2 => {
            let nr = 16usize;
            let mr = 6usize;
            // kc so that kc*nr*4 fits comfortably in L1D (leave ~25% headroom)
            let kc_from_l1 = ((caches.l1d_bytes * 3 / 4) / (nr * 4)).max(64);
            // Prefer multiples of 8 for efficient inner loops
            let kc = floor_to_multiple(kc_from_l1.min(512), 8).max(64);
            // mc so that kc*mc*4 <= ~3/4 L2, keep multiple of mr
            let mc_from_l2 = ((caches.l2_bytes * 3 / 4) / (kc * 4)).max(mr);
            let mc = floor_to_multiple(mc_from_l2.min(240), mr).max(mr);
            // nc so that kc*nc*4 <= ~7/8 L3 (or reasonable default), multiple of nr
            let l3 = caches.l3_bytes.max(1);
            let nc_from_l3 = ((l3 * 7 / 8) / (kc * 4)).max(nr * 8);
            let nc = floor_to_multiple(nc_from_l3.min(2048), nr).max(nr * 8);

            BlockingParams { nc, kc, mc, nr, mr }
        }
        SimdLevel::Sse2 => {
            let nr = 8usize;
            let mr = 4usize;
            let kc_from_l1 = ((caches.l1d_bytes * 3 / 4) / (nr * 4)).max(64);
            let kc = floor_to_multiple(kc_from_l1.min(256), 4).max(64);
            let mc_from_l2 = ((caches.l2_bytes * 3 / 4) / (kc * 4)).max(mr);
            let mc = floor_to_multiple(mc_from_l2.min(128), mr).max(mr);
            let l3 = caches.l3_bytes.max(1);
            let nc_from_l3 = ((l3 * 7 / 8) / (kc * 4)).max(nr * 8);
            let nc = floor_to_multiple(nc_from_l3.min(1024), nr).max(nr * 4);
            BlockingParams { nc, kc, mc, nr, mr }
        }
        SimdLevel::Scalar => BlockingParams {
            nc: 256,
            kc: 64,
            mc: 32,
            nr: 4,
            mr: 4,
        },
    }
}

fn detect_cpu_caches() -> Option<CpuCaches> {
    #[cfg(target_arch = "x86_64")]
    {
        // CPUID leaf 4 provides deterministic cache parameters
        let mut i = 0u32;
        let mut l1d = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        loop {
            let r = unsafe { __cpuid_count(4, i) };
            let cache_type = r.eax & 0x1F; // 0=none,1=data,2=instruction,3=unified
            if cache_type == 0 {
                break;
            }
            let level = (r.eax >> 5) & 0x7; // 1=L1,2=L2,3=L3
            let line_size = ((r.ebx & 0xFFF) + 1) as usize; // 11:0
            let partitions = (((r.ebx >> 12) & 0x3FF) + 1) as usize; // 21:12
            let ways = (((r.ebx >> 22) & 0x3FF) + 1) as usize; // 31:22
            let sets = (r.ecx + 1) as usize;
            let size_bytes = line_size * partitions * ways * sets;
            match (cache_type, level) {
                (1, 1) => l1d = size_bytes,        // L1 data
                (3, 2) => l2 = size_bytes,         // L2 unified
                (3, 3) => l3 = size_bytes,         // L3 unified
                (1, 2) => l2 = l2.max(size_bytes), // some CPUs separate D/I, unify by max
                (1, 3) => l3 = l3.max(size_bytes),
                _ => {}
            }
            i += 1;
        }
        if l1d > 0 {
            // Reasonable fallbacks if some levels missing
            if l2 == 0 {
                l2 = 256 * 1024;
            }
            if l3 == 0 {
                l3 = 8 * 1024 * 1024;
            }
            return Some(CpuCaches {
                l1d_bytes: l1d,
                l2_bytes: l2,
                l3_bytes: l3,
            });
        }
        None
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        None
    }
}

/// Packed panel for matrix A (M x K panel packed for efficient access)
pub struct PackedPanelA {
    /// Packed data tensor (SIMD-aligned)
    tensor: Tensor,
    /// Panel dimensions
    #[allow(dead_code)]
    pub m: usize,
    pub k: usize,
    /// Packing stride (typically mr)
    pub stride: usize,
}

/// Packed panel for matrix B (K x N panel packed for efficient access)
pub struct PackedPanelB {
    /// Packed data tensor (SIMD-aligned)
    tensor: Tensor,
    /// Panel dimensions
    pub k: usize,
    #[allow(dead_code)]
    pub n: usize,
    /// Packing stride (typically nr)
    pub stride: usize,
}

unsafe impl Send for PackedPanelA {}
unsafe impl Sync for PackedPanelA {}
unsafe impl Send for PackedPanelB {}
unsafe impl Sync for PackedPanelB {}

impl PackedPanelA {
    /// Create a new packed panel A with optimal layout
    pub fn new(m: usize, k: usize) -> Self {
        let params = BlockingParams::get_cached();
        let stride = params.mr;

        // Compute padded dimensions for efficient packing
        let padded_m = m.div_ceil(stride) * stride;
        let total_elems = padded_m * k;

        // Create tensor with explicit 64B alignment for predictable aligned loads
        let tensor = Tensor::new_uninitialized_aligned(vec![total_elems], 64);

        Self {
            tensor,
            m: padded_m,
            k,
            stride,
        }
    }

    /// Pack matrix A data into this panel
    pub unsafe fn pack_from_matrix(
        &mut self,
        src: *const f32,
        src_row_stride: usize,
        actual_m: usize,
        actual_k: usize,
    ) {
        let mr = self.stride;
        let dst = self.tensor.as_mut_ptr();

        // Pack in mr x k blocks for cache efficiency
        for i_block in (0..actual_m).step_by(mr) {
            let block_m = (actual_m - i_block).min(mr);

            for k_idx in 0..actual_k {
                for m_idx in 0..block_m {
                    let src_idx = (i_block + m_idx) * src_row_stride + k_idx;
                    let dst_idx = (i_block / mr) * (mr * self.k) + k_idx * mr + m_idx;
                    *dst.add(dst_idx) = *src.add(src_idx);
                }

                // Zero-pad incomplete blocks
                for m_idx in block_m..mr {
                    let dst_idx = (i_block / mr) * (mr * self.k) + k_idx * mr + m_idx;
                    *dst.add(dst_idx) = 0.0;
                }
            }
        }
    }

    /// Get pointer to packed data
    pub fn as_ptr(&self) -> *const f32 {
        unsafe { self.tensor.as_ptr() }
    }

    /// Pack matrix A data with arbitrary strides into this panel
    ///
    /// Layout assumption for source: element at (i, k) is at
    ///   src + i * src_row_stride + k * src_col_stride
    pub unsafe fn pack_from_matrix_strided(
        &mut self,
        src: *const f32,
        src_row_stride: usize,
        src_col_stride: usize,
        actual_m: usize,
        actual_k: usize,
    ) {
        let mr = self.stride;
        let dst = self.tensor.as_mut_ptr();

        for i_block in (0..actual_m).step_by(mr) {
            let block_m = (actual_m - i_block).min(mr);

            for k_idx in 0..actual_k {
                for m_idx in 0..block_m {
                    let src_idx = (i_block + m_idx) * src_row_stride + k_idx * src_col_stride;
                    let dst_idx = (i_block / mr) * (mr * self.k) + k_idx * mr + m_idx;
                    *dst.add(dst_idx) = *src.add(src_idx);
                }

                // Zero-pad incomplete blocks
                for m_idx in block_m..mr {
                    let dst_idx = (i_block / mr) * (mr * self.k) + k_idx * mr + m_idx;
                    *dst.add(dst_idx) = 0.0;
                }
            }
        }
    }
}

impl PackedPanelB {
    /// Create a new packed panel B with optimal layout
    pub fn new(k: usize, n: usize) -> Self {
        let params = BlockingParams::get_cached();
        let stride = params.nr;

        // Compute padded dimensions for efficient packing
        let padded_n = n.div_ceil(stride) * stride;
        let total_elems = k * padded_n;

        // Create tensor with explicit 64B alignment for predictable aligned loads
        let tensor = Tensor::new_uninitialized_aligned(vec![total_elems], 64);

        Self {
            tensor,
            k,
            n: padded_n,
            stride,
        }
    }

    /// Pack matrix B data into this panel
    pub unsafe fn pack_from_matrix(
        &mut self,
        src: *const f32,
        src_row_stride: usize,
        actual_k: usize,
        actual_n: usize,
    ) {
        let nr = self.stride;
        let dst = self.tensor.as_mut_ptr();

        // Pack in k x nr blocks for cache efficiency
        for j_block in (0..actual_n).step_by(nr) {
            let block_n = (actual_n - j_block).min(nr);

            for k_idx in 0..actual_k {
                for n_idx in 0..block_n {
                    let src_idx = k_idx * src_row_stride + (j_block + n_idx);
                    let dst_idx = (j_block / nr) * (self.k * nr) + k_idx * nr + n_idx;
                    *dst.add(dst_idx) = *src.add(src_idx);
                }

                // Zero-pad incomplete blocks
                for n_idx in block_n..nr {
                    let dst_idx = (j_block / nr) * (self.k * nr) + k_idx * nr + n_idx;
                    *dst.add(dst_idx) = 0.0;
                }
            }
        }
    }

    /// Get pointer to packed data
    pub fn as_ptr(&self) -> *const f32 {
        unsafe { self.tensor.as_ptr() }
    }

    /// Pack matrix B data with arbitrary strides into this panel
    ///
    /// Layout assumption for source: element at (k, j) is at
    ///   src + k * src_row_stride + j * src_col_stride
    pub unsafe fn pack_from_matrix_strided(
        &mut self,
        src: *const f32,
        src_row_stride: usize,
        src_col_stride: usize,
        actual_k: usize,
        actual_n: usize,
    ) {
        let nr = self.stride;
        let dst = self.tensor.as_mut_ptr();

        // Pack in k x nr blocks for cache efficiency
        for j_block in (0..actual_n).step_by(nr) {
            let block_n = (actual_n - j_block).min(nr);

            for k_idx in 0..actual_k {
                for n_idx in 0..block_n {
                    let src_idx = k_idx * src_row_stride + (j_block + n_idx) * src_col_stride;
                    let dst_idx = (j_block / nr) * (self.k * nr) + k_idx * nr + n_idx;
                    *dst.add(dst_idx) = *src.add(src_idx);
                }

                // Zero-pad incomplete blocks
                for n_idx in block_n..nr {
                    let dst_idx = (j_block / nr) * (self.k * nr) + k_idx * nr + n_idx;
                    *dst.add(dst_idx) = 0.0;
                }
            }
        }
    }
}

/// Panel cache for reusing packed panels across batched operations
pub struct PanelCache {
    /// Cached A panels by (m, k) dimensions
    a_panels: std::collections::HashMap<(usize, usize), PackedPanelA>,
    /// Cached B panels by (k, n) dimensions
    b_panels: std::collections::HashMap<(usize, usize), PackedPanelB>,
    /// Maximum cache size (number of panels)
    max_cache_size: usize,
    /// LRU order for A panels (most-recently used at the back)
    a_lru: std::collections::VecDeque<(usize, usize)>,
    /// LRU order for B panels (most-recently used at the back)
    b_lru: std::collections::VecDeque<(usize, usize)>,
}

impl PanelCache {
    /// Create new panel cache with specified maximum size
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            a_panels: std::collections::HashMap::new(),
            b_panels: std::collections::HashMap::new(),
            max_cache_size,
            a_lru: std::collections::VecDeque::new(),
            b_lru: std::collections::VecDeque::new(),
        }
    }

    /// Get or create packed panel A
    pub fn get_or_create_panel_a(&mut self, m: usize, k: usize) -> &mut PackedPanelA {
        let key = (m, k);

        if !self.a_panels.contains_key(&key) {
            // Evict oldest entries if cache is full
            if self.a_panels.len() >= self.max_cache_size {
                // True LRU eviction
                while let Some(old_key) = self.a_lru.pop_front() {
                    if self.a_panels.remove(&old_key).is_some() {
                        break;
                    }
                }
            }

            self.a_panels.insert(key, PackedPanelA::new(m, k));
        }
        // Update LRU order: move key to back
        if let Some(pos) = self.a_lru.iter().position(|&x| x == key) {
            self.a_lru.remove(pos);
        }
        self.a_lru.push_back(key);

        self.a_panels.get_mut(&key).unwrap()
    }

    /// Get or create packed panel B
    pub fn get_or_create_panel_b(&mut self, k: usize, n: usize) -> &mut PackedPanelB {
        let key = (k, n);

        if !self.b_panels.contains_key(&key) {
            // Evict oldest entries if cache is full
            if self.b_panels.len() >= self.max_cache_size {
                // True LRU eviction
                while let Some(old_key) = self.b_lru.pop_front() {
                    if self.b_panels.remove(&old_key).is_some() {
                        break;
                    }
                }
            }

            self.b_panels.insert(key, PackedPanelB::new(k, n));
        }
        // Update LRU order: move key to back
        if let Some(pos) = self.b_lru.iter().position(|&x| x == key) {
            self.b_lru.remove(pos);
        }
        self.b_lru.push_back(key);

        self.b_panels.get_mut(&key).unwrap()
    }

    /// Get cache statistics
    #[cfg(test)]
    pub fn stats(&self) -> (usize, usize) {
        (self.a_panels.len(), self.b_panels.len())
    }
}

thread_local! {
    static PANEL_CACHE: std::cell::RefCell<PanelCache> = std::cell::RefCell::new(PanelCache::new(16));
}

/// Get access to thread-local panel cache
pub fn with_panel_cache<F, R>(f: F) -> R
where
    F: FnOnce(&mut PanelCache) -> R,
{
    PANEL_CACHE.with(|cache| f(&mut cache.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blocking_params() {
        let params = BlockingParams::for_simd_level(SimdLevel::Avx2);
        assert_eq!(params.nr, 16);
        assert_eq!(params.mr, 6);
        assert!(params.kc > 0);
        assert!(params.mc > 0);
    }

    #[test]
    fn test_packed_panel_a_creation() {
        let panel = PackedPanelA::new(64, 32);
        assert_eq!(panel.k, 32);
        assert!(panel.m >= 64); // May be padded
        assert!(panel.stride > 0);
    }

    #[test]
    fn test_packed_panel_b_creation() {
        let panel = PackedPanelB::new(32, 64);
        assert_eq!(panel.k, 32);
        assert!(panel.n >= 64); // May be padded
        assert!(panel.stride > 0);
    }

    #[test]
    fn test_panel_cache() {
        let mut cache = PanelCache::new(2);

        // Create first panel
        let _panel1 = cache.get_or_create_panel_a(64, 32);
        assert_eq!(cache.stats().0, 1);

        // Create second panel
        let _panel2 = cache.get_or_create_panel_a(128, 64);
        assert_eq!(cache.stats().0, 2);

        // Create third panel - should evict first
        let _panel3 = cache.get_or_create_panel_a(256, 128);
        assert_eq!(cache.stats().0, 2);
    }

    #[test]
    fn test_panel_packing() {
        let mut panel = PackedPanelA::new(4, 4);

        // Create test matrix
        let src_data = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        unsafe {
            panel.pack_from_matrix(src_data.as_ptr(), 4, 4, 4);

            // Verify packing worked (basic sanity check)
            let packed_data = panel.as_ptr();
            assert!(!packed_data.is_null());
        }
    }
}
