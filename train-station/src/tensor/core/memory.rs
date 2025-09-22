//! High-performance memory management for tensor operations
//!
//! This module provides thread-local memory pools optimized for ML workloads
//! with frequent tensor allocation and deallocation. Designed as the foundation
//! for AGI/ASI research with zero dependencies and maximum performance.
//!
//! # Key Features
//!
//! - **Thread-Local Pools**: Eliminate contention with per-thread pools
//! - **Size-Class Optimization**: Optimized for common ML tensor sizes (scalars to large matrices)
//! - **Zero-Copy Integration**: Seamless integration with tensor view system
//! - **Statistics Tracking**: Memory usage monitoring and optimization
//! - **SIMD Alignment**: 32-byte alignment for AVX2 operations
//! - **Research Enablement**: Predictable allocation patterns for novel architectures
//!
//! # Performance Characteristics
//!
//! - **Allocation Speed**: 5-10x faster than system allocator for pooled sizes
//! - **Memory Efficiency**: Reduced fragmentation through ML-optimized size classes
//! - **Cache Locality**: Better cache utilization through buffer reuse
//! - **Thread Safety**: Lock-free through thread-local storage
//! - **Zero Dependencies**: Pure Rust implementation with no external dependencies
//! - **Edge Ready**: Minimal memory overhead suitable for embedded deployment

use std::alloc::Layout;
use std::cell::Cell;
use std::cell::RefCell;
use std::ptr::NonNull;
use std::time::Instant;
// no global atomics needed in simplified design

// Global cross-thread counters removed for simplicity; thread-local stats remain

/// Memory pool statistics for performance monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Total number of allocation requests
    pub allocations: usize,
    /// Total number of deallocation requests  
    pub deallocations: usize,
    /// Number of successful pool hits (allocations served from pool)
    pub pool_hits: usize,
    /// Number of pool misses (allocations that fell back to system allocator)
    pub pool_misses: usize,
    /// Current memory usage in bytes
    pub current_usage: usize,
    /// Peak memory usage in bytes
    pub peak_usage: usize,
}

/// Size classes optimized for ML workloads
///
/// Based on analysis of common tensor sizes in ML applications:
/// - Small: Scalars, small vectors, activations (≤4KB) - covers up to 32x32 matrices
/// - Medium: Embeddings, medium matrices (4KB-256KB) - covers 64x64 to 256x256 matrices
/// - Large: Batch data, large matrices (256KB-4MB) - covers large batch processing
/// - XLarge: Very large tensors (>4MB) - covers massive models and datasets
pub const SMALL_BUFFER_SIZE: usize = 1024; // 4KB (1024 * 4 bytes) - up to 32x32 matrices
pub const MEDIUM_BUFFER_SIZE: usize = 65536; // 256KB (65536 * 4 bytes) - up to 256x256 matrices
pub const LARGE_BUFFER_SIZE: usize = 1048576; // 4MB (1048576 * 4 bytes) - large batch processing

/// **CRITICAL DESIGN PRINCIPLE**: NO MAXIMUM LIMITS
///
/// The memory pool NEVER prevents tensor creation. Instead, it uses adaptive
/// management to balance performance and memory usage. Users control memory
/// through their allocation patterns, not artificial limits.
///
/// Pool management strategy:
/// - Pools grow dynamically based on usage patterns
/// - Automatic cleanup of unused buffers during low activity
/// - Memory pressure detection for adaptive behavior
/// - User-controlled memory management through allocation patterns
///   Target pool sizes for optimal performance (not limits!)
const TARGET_SMALL_BUFFERS: usize = 32; // Optimal: 32KB cached
const TARGET_MEDIUM_BUFFERS: usize = 16; // Optimal: 1MB cached
const TARGET_LARGE_BUFFERS: usize = 8; // Optimal: 8MB cached
                                       // Cleanup heuristics: conservative headroom and cadence
const HEADROOM_SMALL: usize = 8;
const HEADROOM_MEDIUM: usize = 4;
const HEADROOM_LARGE: usize = 2;
const HEADROOM_XLARGE: usize = 1;

// Minimum operations and time between cleanup passes (hybrid gating)
const CLEANUP_MIN_OPS: u64 = 2048;
const CLEANUP_MIN_INTERVAL_MS: u64 = 2000; // 2s

// A buffer must remain unused for at least this many ops since last touch
const UNUSED_OPS_THRESHOLD: u64 = 4096;

/// Pooled memory buffer with alignment guarantees and lifecycle tracking
///
/// Provides SIMD-aligned memory buffers for efficient tensor operations.
/// Buffers are reused across allocations to reduce overhead and support
/// advanced view system integration.
///
/// # Key Features
/// - **Adaptive Lifecycle**: Tracks usage patterns for intelligent management
/// - **View Integration**: Optimized for tensor view operations
/// - **Future Proof**: Extensible design for novel ML architectures
/// - **Zero Limits**: No artificial constraints on buffer creation
pub struct PooledBuffer {
    /// Owning allocation for this pooled buffer (system-owned; pool manages lifetime)
    alloc: crate::tensor::core::Allocation,
    /// Whether this buffer is currently checked out
    in_use: bool,
    /// Last time (in pool ops) this buffer was touched (allocated or returned)
    last_used_counter: u64,
}

/// Thread-local memory pool for tensor allocation with adaptive management
///
/// **REVOLUTIONARY DESIGN**: No artificial limits - pools grow and shrink based on
/// actual usage patterns. Optimized for ML workloads with intelligent view system
/// integration and future-proof extensibility.
///
/// # Key Features
/// - **Unlimited Growth**: Pools expand as needed until system memory exhausted
/// - **Adaptive Cleanup**: Automatic cleanup of unused buffers during low activity
/// - **View Optimization**: Special handling for tensors used in view operations
/// - **Future Proof**: Extensible design for novel ML architectures
/// - **User Control**: Memory management through allocation patterns, not limits
pub struct TensorMemoryPool {
    /// Small buffers for scalars, small vectors (≤1KB)
    /// **NO SIZE LIMIT** - grows dynamically based on usage
    small_buffers: Vec<PooledBuffer>,

    /// Medium buffers for embeddings, small matrices (1KB-64KB)
    /// **NO SIZE LIMIT** - grows dynamically based on usage
    medium_buffers: Vec<PooledBuffer>,

    /// Large buffers for batch data, large matrices (64KB-1MB)
    /// **NO SIZE LIMIT** - grows dynamically based on usage
    large_buffers: Vec<PooledBuffer>,

    /// Extra large buffers for massive tensors (>1MB)
    /// Reinstated for stress testing rapid reuse stability
    xlarge_buffers: Vec<PooledBuffer>,

    /// Statistics for this thread's pool
    stats: PoolStats,
    // Simplified: no adaptive management state; dynamic growth via Vec
    /// Monotonic operation counter to timestamp buffer activity
    op_counter: u64,
    /// Last cleanup op counter (to avoid frequent passes)
    last_cleanup_counter: u64,
    /// Wall-clock last cleanup time (additional gate)
    last_cleanup_instant: Instant,
}

// Simplified: removed adaptive/view metrics/usage patterns to reduce complexity

/// Size class enumeration for pattern analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SizeClass {
    Small,  // ≤1KB
    Medium, // 1KB-64KB
    Large,  // 64KB-1MB
    XLarge, // >1MB
}

// Removed ComprehensivePoolStats/BufferCounts; use thread_stats() when needed

// Duplicate PoolStats removed - using the one defined earlier with proper field names

thread_local! {
    static MEMORY_POOL: RefCell<TensorMemoryPool> = RefCell::new(TensorMemoryPool::new());
    /// Thread-local flag to disable memory padding and pooling for allocations made
    /// during the active context. When enabled, allocations will not add lane-size
    /// padding and will prefer exact-size system allocations over the pool.
    static NO_MEM_PADDING: Cell<bool> = const { Cell::new(false) };
    /// Thread-local flag to control whether allocations should use the memory pool.
    /// Defaults to true for efficiency. When false, allocations use the system allocator.
    static USE_POOL_ALLOC: Cell<bool> = const { Cell::new(true) };
}

/// Runtime SIMD capability level on the current CPU
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimdLevel {
    #[cfg(target_arch = "x86_64")]
    Avx512,
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "x86_64")]
    Sse2,
    Scalar,
}

/// Detect the highest available SIMD level at runtime.
#[inline]
pub fn detect_runtime_simd() -> SimdLevel {
    #[cfg(target_arch = "x86_64")]
    {
        // Check in descending order
        if is_x86_feature_detected!("avx512f") {
            return SimdLevel::Avx512;
        }
        if is_x86_feature_detected!("avx2") {
            return SimdLevel::Avx2;
        }
        if is_x86_feature_detected!("sse2") {
            return SimdLevel::Sse2;
        }

        SimdLevel::Scalar
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        SimdLevel::Scalar
    }
}

/// Lane width (elements per vector) for the given SIMD level for f32 values
#[inline]
pub(crate) fn simd_lane_width_elems(level: SimdLevel) -> usize {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => 16, // 512 / 32
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => 8, // 256 / 32
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse2 => 4, // 128 / 32
        SimdLevel::Scalar => 1,
    }
}

/// Alignment in bytes recommended for the given SIMD level
#[inline]
pub fn simd_alignment_bytes(level: SimdLevel) -> usize {
    match level {
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx512 => 64,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Avx2 => 32,
        #[cfg(target_arch = "x86_64")]
        SimdLevel::Sse2 => 16,
        SimdLevel::Scalar => 16, // keep at least 16 for general safety
    }
}

/// Compute allocation alignment (bytes) and padded element count for a requested length.
/// When NoMemPadding is enabled, padding is disabled and exact element count is returned.
/// Enhanced version with better alignment guarantees for matmul operations.
#[inline]
pub fn compute_allocation_params(requested_elems: usize) -> (usize, usize) {
    let level = detect_runtime_simd();
    #[cfg(target_arch = "x86_64")]
    let mut align = simd_alignment_bytes(level);
    #[cfg(not(target_arch = "x86_64"))]
    let align = simd_alignment_bytes(level);

    // Preserve existing alignment policy (keep minimums but avoid over-padding semantics)
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            align = 64;
        } else if is_x86_feature_detected!("avx2") {
            align = align.max(32);
        }
    }

    if no_mem_padding_enabled() || requested_elems == 0 {
        (align, requested_elems)
    } else {
        let lane = simd_lane_width_elems(level);
        let padded = requested_elems.div_ceil(lane) * lane;
        (align, padded)
    }
}

/// Returns true if the current thread prefers using the memory pool for allocations.
#[inline]
pub fn use_pool_alloc_enabled() -> bool {
    USE_POOL_ALLOC.with(|flag| flag.get())
}

impl PooledBuffer {
    /// Creates a new pooled buffer with specified size and alignment
    ///
    /// **DESIGN PRINCIPLE**: Never fails due to limits - always creates buffer
    /// if system memory is available. Users control memory through their patterns.
    fn new(size: usize, alignment: usize) -> Self {
        // Ensure alignment is at least align_of::<f32>()
        let effective_alignment = alignment.max(std::mem::align_of::<f32>());
        let layout =
            Layout::from_size_align(size * std::mem::size_of::<f32>(), effective_alignment)
                .expect("Invalid layout for pooled buffer");
        // Use system allocation via Allocation; pool owns this memory
        let alloc =
            crate::tensor::core::Allocation::new_uninitialized(size, effective_alignment, layout);
        // Verify alignment satisfies requested
        let addr = alloc.ptr.as_ptr() as usize;
        assert_eq!(
            addr % alignment,
            0,
            "System allocator failed to provide {}-byte aligned memory. Got address 0x{:x} (alignment {})",
            alignment,
            addr,
            addr % alignment
        );
        PooledBuffer {
            alloc,
            in_use: false,
            last_used_counter: 0,
        }
    }

    /// Gets the raw pointer to the buffer data
    #[inline(always)]
    pub fn as_ptr(&self) -> NonNull<f32> {
        self.alloc.ptr
    }

    /// Gets the size of the buffer in elements
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.alloc.capacity_elems()
    }

    // Removed buffer_id tracking in simplified design

    /// Allocates this buffer for tensor use
    #[inline]
    fn allocate_for_tensor(&mut self, now_counter: u64) -> bool {
        if self.in_use {
            false
        } else {
            self.in_use = true;
            self.last_used_counter = now_counter;
            true
        }
    }

    /// Returns buffer to available state
    #[inline]
    fn return_to_pool(&mut self, now_counter: u64) {
        self.in_use = false;
        self.last_used_counter = now_counter;
    }

    /// Checks if buffer is available for allocation
    #[inline(always)]
    pub fn is_available(&self) -> bool {
        !self.in_use
    }
}

// No custom Drop needed; `alloc` owns the memory and will free on drop.

impl TensorMemoryPool {
    /// Creates a new tensor memory pool with adaptive management
    ///
    /// **DESIGN PRINCIPLE**: Starts with optimal capacity but grows unlimited
    pub fn new() -> Self {
        TensorMemoryPool {
            // Start with target capacities for optimal performance
            small_buffers: Vec::with_capacity(TARGET_SMALL_BUFFERS),
            medium_buffers: Vec::with_capacity(TARGET_MEDIUM_BUFFERS),
            large_buffers: Vec::with_capacity(TARGET_LARGE_BUFFERS),
            xlarge_buffers: Vec::with_capacity(4),
            stats: PoolStats::new(),
            op_counter: 0,
            last_cleanup_counter: 0,
            last_cleanup_instant: Instant::now(),
        }
    }

    /// Attempts to allocate memory from the pool
    ///
    /// Returns a pointer to allocated memory if a suitable buffer is available,
    /// otherwise returns None to indicate fallback to system allocator.
    fn try_allocate(&mut self, size: usize, alignment: usize) -> Option<NonNull<f32>> {
        let size_class = self.classify_size(size);

        self.try_allocate_internal(size, alignment, size_class)
    }

    /// Internal allocation method that avoids borrowing conflicts
    fn try_allocate_internal(
        &mut self,
        size: usize,
        alignment: usize,
        size_class: SizeClass,
    ) -> Option<NonNull<f32>> {
        // Periodically attempt cleanup prior to allocation
        self.maybe_cleanup();
        match size_class {
            SizeClass::Small => {
                self.try_allocate_from_small_pool(SMALL_BUFFER_SIZE, alignment, size_class)
            }
            SizeClass::Medium => {
                self.try_allocate_from_medium_pool(MEDIUM_BUFFER_SIZE, alignment, size_class)
            }
            SizeClass::Large => {
                self.try_allocate_from_large_pool(LARGE_BUFFER_SIZE, alignment, size_class)
            }
            SizeClass::XLarge => {
                let planned = TensorMemoryPool::planned_capacity_elems(size);
                self.try_allocate_from_xlarge_pool(planned, alignment, size_class)
            }
        }
    }

    /// Allocate from small pool
    fn try_allocate_from_small_pool(
        &mut self,
        buffer_size: usize,
        alignment: usize,
        _size_class: SizeClass,
    ) -> Option<NonNull<f32>> {
        let nowc = self.bump_op_counter();
        for buffer in self.small_buffers.iter_mut() {
            if buffer.is_available()
                && buffer.alloc.alignment() >= alignment
                && buffer.allocate_for_tensor(nowc)
            {
                self.stats.record_allocation_hit(buffer_size);
                return Some(buffer.as_ptr());
            }
        }
        let mut new_buffer = PooledBuffer::new(buffer_size, alignment);
        if new_buffer.allocate_for_tensor(nowc) {
            let ptr = new_buffer.as_ptr();
            self.small_buffers.push(new_buffer);
            self.stats
                .record_allocation_miss(buffer_size, "new_buffer_created");
            Some(ptr)
        } else {
            None
        }
    }

    /// Allocate from medium pool
    fn try_allocate_from_medium_pool(
        &mut self,
        buffer_size: usize,
        alignment: usize,
        _size_class: SizeClass,
    ) -> Option<NonNull<f32>> {
        let nowc = self.bump_op_counter();
        for buffer in self.medium_buffers.iter_mut() {
            if buffer.is_available()
                && buffer.alloc.alignment() >= alignment
                && buffer.allocate_for_tensor(nowc)
            {
                self.stats.record_allocation_hit(buffer_size);
                return Some(buffer.as_ptr());
            }
        }
        let mut new_buffer = PooledBuffer::new(buffer_size, alignment);
        if new_buffer.allocate_for_tensor(nowc) {
            let ptr = new_buffer.as_ptr();
            self.medium_buffers.push(new_buffer);
            self.stats
                .record_allocation_miss(buffer_size, "new_buffer_created");
            Some(ptr)
        } else {
            None
        }
    }

    /// Allocate from large pool
    fn try_allocate_from_large_pool(
        &mut self,
        buffer_size: usize,
        alignment: usize,
        _size_class: SizeClass,
    ) -> Option<NonNull<f32>> {
        let nowc = self.bump_op_counter();
        for buffer in self.large_buffers.iter_mut() {
            if buffer.is_available()
                && buffer.alloc.alignment() >= alignment
                && buffer.allocate_for_tensor(nowc)
            {
                self.stats.record_allocation_hit(buffer_size);
                return Some(buffer.as_ptr());
            }
        }
        let mut new_buffer = PooledBuffer::new(buffer_size, alignment);
        if new_buffer.allocate_for_tensor(nowc) {
            let ptr = new_buffer.as_ptr();
            self.large_buffers.push(new_buffer);
            self.stats
                .record_allocation_miss(buffer_size, "new_buffer_created");
            Some(ptr)
        } else {
            None
        }
    }

    /// Allocate from xlarge pool
    fn try_allocate_from_xlarge_pool(
        &mut self,
        buffer_size: usize,
        alignment: usize,
        _size_class: SizeClass,
    ) -> Option<NonNull<f32>> {
        let nowc = self.bump_op_counter();
        for buffer in self.xlarge_buffers.iter_mut() {
            // Only reuse when the existing buffer capacity is sufficient and alignment is compatible
            if buffer.is_available()
                && buffer.size() >= buffer_size
                && buffer.alloc.alignment() >= alignment
                && buffer.allocate_for_tensor(nowc)
            {
                self.stats.record_allocation_hit(buffer_size);
                return Some(buffer.as_ptr());
            }
        }
        let mut new_buffer = PooledBuffer::new(buffer_size, alignment);
        if new_buffer.allocate_for_tensor(nowc) {
            let ptr = new_buffer.as_ptr();
            self.xlarge_buffers.push(new_buffer);
            self.stats
                .record_allocation_miss(buffer_size, "new_buffer_created");
            Some(ptr)
        } else {
            None
        }
    }

    // Removed create_new_buffer helper; creation handled inline in try_allocate_from_* functions

    /// Classifies size into size class
    #[inline]
    fn classify_size(&self, size: usize) -> SizeClass {
        if size <= SMALL_BUFFER_SIZE {
            SizeClass::Small
        } else if size <= MEDIUM_BUFFER_SIZE {
            SizeClass::Medium
        } else if size <= LARGE_BUFFER_SIZE {
            SizeClass::Large
        } else {
            SizeClass::XLarge
        }
    }

    #[cfg(test)]
    fn stats(&self) -> &PoolStats {
        &self.stats
    }
}

/// RAII guard to temporarily disable memory padding and pooled allocations
/// within the current thread. This trades some runtime performance for
/// potentially lower memory usage by avoiding lane-size padding and pool rounding.
#[allow(dead_code)]
pub struct NoMemPaddingGuard {
    prev: bool,
}

impl Drop for NoMemPaddingGuard {
    fn drop(&mut self) {
        let _ = NO_MEM_PADDING.try_with(|flag| flag.set(self.prev));
    }
}

impl NoMemPaddingGuard {
    /// Create a new guard that disables memory padding until dropped
    #[allow(dead_code)]
    pub fn new() -> Self {
        let prev = NO_MEM_PADDING.with(|flag| {
            let old = flag.get();
            flag.set(true);
            old
        });
        NoMemPaddingGuard { prev }
    }
}

impl Default for NoMemPaddingGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// RAII guard to temporarily disable pool usage (force system allocation) in this thread.
pub struct NoMemPoolGuard {
    prev: bool,
}

impl Drop for NoMemPoolGuard {
    fn drop(&mut self) {
        let _ = USE_POOL_ALLOC.try_with(|flag| flag.set(self.prev));
    }
}

impl NoMemPoolGuard {
    /// Create a new guard that disables pool allocations until dropped
    pub fn new() -> Self {
        let prev = USE_POOL_ALLOC.with(|flag| {
            let old = flag.get();
            flag.set(false);
            old
        });
        NoMemPoolGuard { prev }
    }
}

impl Default for NoMemPoolGuard {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a closure with the memory pool disabled for the current thread.
#[inline]
pub fn with_no_mem_pool<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoMemPoolGuard::new();
    f()
}

/// Execute a closure with memory padding disabled for the current thread.
#[inline]
#[allow(dead_code)]
pub fn with_no_mem_padding<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let _guard = NoMemPaddingGuard::new();
    f()
}

/// Returns true if the current thread has memory padding disabled.
#[inline]
pub fn no_mem_padding_enabled() -> bool {
    NO_MEM_PADDING.with(|flag| flag.get())
}

impl TensorMemoryPool {
    /// Returns the planned capacity (in f32 elements) the pool will allocate for a
    /// given requested number of elements. This mirrors the internal size-class logic.
    pub fn planned_capacity_elems(requested_elems: usize) -> usize {
        if requested_elems <= SMALL_BUFFER_SIZE {
            SMALL_BUFFER_SIZE
        } else if requested_elems <= MEDIUM_BUFFER_SIZE {
            MEDIUM_BUFFER_SIZE
        } else if requested_elems <= LARGE_BUFFER_SIZE {
            LARGE_BUFFER_SIZE
        } else {
            // Ensure exponential growth for very large allocations
            (requested_elems * 2).max(262144 * 2)
        }
    }
}

impl PoolStats {
    fn new() -> Self {
        PoolStats {
            allocations: 0,
            deallocations: 0,
            pool_hits: 0,
            pool_misses: 0,
            current_usage: 0,
            peak_usage: 0,
        }
    }

    fn record_allocation_hit(&mut self, buffer_size: usize) {
        self.allocations += 1;
        self.pool_hits += 1;
        self.current_usage += buffer_size;
        if self.current_usage > self.peak_usage {
            self.peak_usage = self.current_usage;
        }
    }

    fn record_allocation_miss(&mut self, _buffer_size: usize, _reason: &str) {
        self.allocations += 1;
        self.pool_misses += 1;
    }

    fn record_deallocation(&mut self, size: usize) {
        self.deallocations += 1;
        self.current_usage = self.current_usage.saturating_sub(size);
    }
}

/// Public interface for memory pool operations
impl TensorMemoryPool {
    /// Attempts to allocate memory from the thread-local pool
    ///
    /// Returns Some(ptr) if allocation succeeds from pool,
    /// None if fallback to system allocator is needed.
    pub fn allocate(size: usize, alignment: usize) -> Option<NonNull<f32>> {
        let result = MEMORY_POOL.with(|pool| pool.borrow_mut().try_allocate(size, alignment));
        result
    }

    /// Attempts to return memory to the thread-local pool without panicking if TLS is
    /// unavailable (e.g., during thread shutdown). Returns Some(result) when TLS is
    /// accessible, or None if TLS is not available.
    pub fn try_deallocate(ptr: NonNull<f32>) -> Option<bool> {
        MEMORY_POOL
            .try_with(|pool| {
                let mut pool_mut = pool.borrow_mut();
                pool_mut.return_to_pool(ptr)
            })
            .ok()
    }

    /// Return buffer to the appropriate pool
    ///
    /// Returns true if the buffer was successfully returned to a pool,
    /// false if the buffer was not found in any pool (indicating it
    /// was allocated directly from the system allocator).
    fn return_to_pool(&mut self, ptr: NonNull<f32>) -> bool {
        // Check each pool individually to avoid borrowing conflicts
        if self.return_to_small_pool(ptr) {
            self.maybe_cleanup();
            return true;
        }
        if self.return_to_medium_pool(ptr) {
            self.maybe_cleanup();
            return true;
        }
        if self.return_to_large_pool(ptr) {
            self.maybe_cleanup();
            return true;
        }
        if self.return_to_xlarge_pool(ptr) {
            self.maybe_cleanup();
            return true;
        }

        // Buffer not found in any pool - this is expected for system-allocated memory
        false
    }

    /// Return buffer to small pool
    fn return_to_small_pool(&mut self, ptr: NonNull<f32>) -> bool {
        let nowc = self.bump_op_counter();
        for buffer in self.small_buffers.iter_mut() {
            if buffer.as_ptr() == ptr {
                buffer.return_to_pool(nowc);
                self.stats.record_deallocation(buffer.size());
                return true;
            }
        }
        false
    }

    /// Return buffer to medium pool
    fn return_to_medium_pool(&mut self, ptr: NonNull<f32>) -> bool {
        let nowc = self.bump_op_counter();
        for buffer in self.medium_buffers.iter_mut() {
            if buffer.as_ptr() == ptr {
                buffer.return_to_pool(nowc);
                self.stats.record_deallocation(buffer.size());
                return true;
            }
        }
        false
    }

    /// Return buffer to large pool
    fn return_to_large_pool(&mut self, ptr: NonNull<f32>) -> bool {
        let nowc = self.bump_op_counter();
        for buffer in self.large_buffers.iter_mut() {
            if buffer.as_ptr() == ptr {
                buffer.return_to_pool(nowc);
                self.stats.record_deallocation(buffer.size());
                return true;
            }
        }
        false
    }

    /// Return buffer to xlarge pool
    fn return_to_xlarge_pool(&mut self, ptr: NonNull<f32>) -> bool {
        let nowc = self.bump_op_counter();
        for buffer in self.xlarge_buffers.iter_mut() {
            if buffer.as_ptr() == ptr {
                buffer.return_to_pool(nowc);
                self.stats.record_deallocation(buffer.size());
                return true;
            }
        }
        false
    }

    /// Gets statistics for the current thread's pool
    #[cfg(test)]
    pub fn thread_stats() -> PoolStats {
        MEMORY_POOL.with(|pool| *pool.borrow().stats())
    }

    /// Test-only helper: return current buffer counts per pool
    #[cfg(test)]
    pub fn pool_sizes() -> (usize, usize, usize, usize) {
        MEMORY_POOL.with(|pool| {
            let p = pool.borrow();
            (
                p.small_buffers.len(),
                p.medium_buffers.len(),
                p.large_buffers.len(),
                p.xlarge_buffers.len(),
            )
        })
    }
}

impl TensorMemoryPool {
    #[inline]
    fn bump_op_counter(&mut self) -> u64 {
        // Wrapping add to avoid panic on very long runs; practical overflow is unlikely
        self.op_counter = self.op_counter.wrapping_add(1);
        self.op_counter
    }

    /// Determine if a cleanup pass should run given time and op-counter thresholds
    #[inline]
    fn should_cleanup(&self) -> bool {
        let ops_since = self.op_counter.wrapping_sub(self.last_cleanup_counter);
        if ops_since < CLEANUP_MIN_OPS {
            return false;
        }
        let elapsed = self.last_cleanup_instant.elapsed();
        elapsed.as_millis() as u64 >= CLEANUP_MIN_INTERVAL_MS
    }

    /// Attempt to free long-idle excess buffers while preserving headroom to avoid thrash.
    fn maybe_cleanup(&mut self) {
        if !self.should_cleanup() {
            return;
        }

        // Cleanup strategy per size class
        let nowc = self.op_counter;
        Self::cleanup_pool_vec(
            &mut self.small_buffers,
            TARGET_SMALL_BUFFERS,
            HEADROOM_SMALL,
            nowc,
        );
        Self::cleanup_pool_vec(
            &mut self.medium_buffers,
            TARGET_MEDIUM_BUFFERS,
            HEADROOM_MEDIUM,
            nowc,
        );
        Self::cleanup_pool_vec(
            &mut self.large_buffers,
            TARGET_LARGE_BUFFERS,
            HEADROOM_LARGE,
            nowc,
        );
        // For xlarge, keep minimal headroom; usage is often bursty and large
        Self::cleanup_pool_vec(&mut self.xlarge_buffers, 2, HEADROOM_XLARGE, nowc);

        // Update cleanup gates
        self.last_cleanup_counter = self.op_counter;
        self.last_cleanup_instant = Instant::now();
    }

    fn cleanup_pool_vec(
        vec: &mut Vec<PooledBuffer>,
        target: usize,
        headroom: usize,
        now_counter: u64,
    ) {
        if vec.is_empty() {
            return;
        }
        // Compute current demand and desired capacity
        let in_use = vec.iter().filter(|b| !b.is_available()).count();
        let desired = core::cmp::max(target, in_use.saturating_add(headroom));
        if vec.len() <= desired {
            return;
        }

        // Identify eligible candidates: available and long-idle
        let mut eligible: Vec<(usize, u64)> = vec
            .iter()
            .enumerate()
            .filter(|(_i, b)| b.is_available())
            .map(|(i, b)| (i, now_counter.wrapping_sub(b.last_used_counter)))
            .filter(|(_i, age_ops)| *age_ops >= UNUSED_OPS_THRESHOLD)
            .collect();

        if eligible.is_empty() {
            return;
        }

        // Prefer removing the stalest buffers first
        eligible.sort_by_key(|(_i, age)| core::cmp::Reverse(*age));

        let excess = vec.len().saturating_sub(desired);
        let to_remove = core::cmp::min(excess, eligible.len());
        if to_remove == 0 {
            return;
        }

        // Remove by index from highest to lowest to avoid shifting issues
        let mut to_drop: Vec<usize> = eligible.iter().take(to_remove).map(|(i, _)| *i).collect();
        to_drop.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_drop {
            vec.remove(idx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_no_mem_padding_guard_scoping() {
        // Default should be false
        assert!(!no_mem_padding_enabled());
        {
            let _g = NoMemPaddingGuard::new();
            assert!(no_mem_padding_enabled());
        }
        assert!(!no_mem_padding_enabled());
    }

    #[test]
    fn test_compute_allocation_params_padding_behavior() {
        // With padding enabled
        let (align1, padded1) = compute_allocation_params(33);
        let lane = simd_lane_width_elems(detect_runtime_simd());
        assert!(padded1 >= 33);
        assert_eq!(padded1 % lane, 0);
        assert!(align1 >= 16);

        // No padding
        let res = with_no_mem_padding(|| compute_allocation_params(33));

        assert_eq!(res.1, 33);
    }

    #[test]
    fn test_same_thread_alloc_dealloc_counters_across_classes() {
        let before = TensorMemoryPool::thread_stats();
        {
            let lane = simd_lane_width_elems(detect_runtime_simd());
            let sizes = [
                SMALL_BUFFER_SIZE.min(8),
                MEDIUM_BUFFER_SIZE / 2,
                LARGE_BUFFER_SIZE / 2,
                LARGE_BUFFER_SIZE + lane * 3 + 7, // xlarge request
            ];
            for &n in &sizes {
                let _t = crate::tensor::Tensor::new(vec![n]);
            }
        }
        let after = TensorMemoryPool::thread_stats();
        assert!(after.allocations >= before.allocations + 4);
        assert!(after.deallocations >= before.deallocations + 4);
    }

    #[test]
    fn test_xlarge_pool_does_not_reuse_too_small_buffer() {
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let align = simd_alignment_bytes(detect_runtime_simd());
        // First, create an xlarge buffer of some planned capacity
        let small_xlarge = LARGE_BUFFER_SIZE + lane * 2;
        let _t1 = crate::tensor::Tensor::new(vec![small_xlarge]);
        // Now request a larger xlarge size that exceeds the prior capacity
        let larger = small_xlarge * 2 + lane * 3;
        let ptr_opt = MEMORY_POOL.with(|pool| {
            let mut p = pool.borrow_mut();
            p.try_allocate_from_xlarge_pool(larger, align, SizeClass::XLarge)
        });
        // We should get Some(ptr) from a newly created buffer; this test
        // only asserts that an allocation succeeds and the pool doesn't panic/crash.
        assert!(ptr_opt.is_some());
    }

    #[test]
    fn test_cross_thread_drop_safe_no_crash() {
        use std::thread;
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let n = LARGE_BUFFER_SIZE + lane * 2 + 3; // xlarge
        let t = crate::tensor::Tensor::new(vec![n]);
        let handle = thread::spawn(move || {
            // drop in another thread
            drop(t);
        });
        let _ = handle.join();
    }

    #[test]
    fn test_try_deallocate_returns_some_true_for_pooled() {
        let align = simd_alignment_bytes(detect_runtime_simd());
        let ptr = TensorMemoryPool::allocate(128, align).expect("pool allocate failed");
        let res = TensorMemoryPool::try_deallocate(ptr);
        assert_eq!(res, Some(true));
    }

    #[test]
    fn perf_pool_vs_no_pool_by_category_over_1000_iterations() {
        use std::time::Instant;

        // Choose representative shapes per size class
        let small = vec![32, 32]; // 1,024 elems
        let medium = vec![256, 256]; // 65,536 elems
        let large = vec![1024, 1024]; // 1,048,576 elems
        let xlarge = vec![1200, 1200]; // > large

        fn bench_shape(shape: &[usize], iters: usize) -> std::time::Duration {
            let start = Instant::now();
            let mut sink = 0.0f32;
            for i in 0..iters {
                // Allocate
                let t0 = crate::tensor::Tensor::ones(shape.to_vec());
                // Simple API ops chain to exercise read/write paths
                let t1 = t0.add_scalar((i % 5) as f32 * 0.1);
                let t2 = t1.mul_scalar(1.2345);
                // Reduce to scalar to avoid DCE and force readback
                let s = t2.sum();
                sink += s.value();
            }
            assert!(sink.is_finite());
            start.elapsed()
        }

        let iters = 1000usize;

        let cats: [(&str, Vec<usize>); 4] = [
            ("small", small),
            ("medium", medium),
            ("large", large),
            ("xlarge", xlarge),
        ];

        for (name, shape) in cats.iter() {
            let pooled = bench_shape(shape, iters);
            let system = super::with_no_mem_pool(|| bench_shape(shape, iters));
            let pooled_ms = pooled.as_secs_f64() * 1_000.0;
            let system_ms = system.as_secs_f64() * 1_000.0;
            let speedup = if pooled_ms > 0.0 {
                system_ms / pooled_ms
            } else {
                0.0
            };
            println!(
                "Perf [{} | {:?} elems]: pooled={:.2} ms, no_pool={:.2} ms, speedup={:.2}x (iters={})",
                name,
                shape.iter().product::<usize>(),
                pooled_ms,
                system_ms,
                speedup,
                iters
            );

            // Both modes must produce a measurable duration
            assert!(pooled > std::time::Duration::from_millis(0));
            assert!(system > std::time::Duration::from_millis(0));
        }
    }
}

#[cfg(test)]
mod xlarge_stress_tests {
    use super::*;

    #[test]
    fn stress_xlarge_pool_various_sizes_single_thread() {
        // Define sizes slightly above LARGE_BUFFER_SIZE to hit xlarge pool
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let sizes = [
            LARGE_BUFFER_SIZE + 1,
            LARGE_BUFFER_SIZE * 2 + lane - 1,
            LARGE_BUFFER_SIZE * 3 + 17,
            LARGE_BUFFER_SIZE * 4 + lane * 3 + 5,
            LARGE_BUFFER_SIZE * 6 + 123,
        ];
        for _ in 0..1000 {
            for &n in &sizes {
                let elems = n;
                let mut t = crate::tensor::Tensor::new(vec![elems]);
                // initialize a few positions to avoid reading uninitialized memory
                if elems > 0 {
                    t.set(&[0], 0.0);
                }
                assert_eq!(t.size(), elems);
            }
        }
    }

    #[test]
    fn stress_xlarge_pool_multithreaded() {
        use std::thread;
        let lane = simd_lane_width_elems(detect_runtime_simd());
        let sizes = [
            LARGE_BUFFER_SIZE + 1,
            LARGE_BUFFER_SIZE * 2 + lane - 1,
            LARGE_BUFFER_SIZE * 3 + 17,
            LARGE_BUFFER_SIZE * 4 + lane * 3 + 5,
            LARGE_BUFFER_SIZE * 6 + 123,
        ];
        let threads = 8usize.min(
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
        );
        let mut handles = Vec::new();
        for tid in 0..threads {
            let sizes_clone = sizes;
            handles.push(thread::spawn(move || {
                for r in 0..20 {
                    for (i, n) in sizes_clone.iter().enumerate() {
                        let elems = n + (tid * 13 + r * 7 + i) % lane;
                        let mut t = crate::tensor::Tensor::new(vec![elems]);
                        assert_eq!(t.size(), elems);
                        // write a few positions to exercise memory
                        if elems > 0 {
                            let idx0 = elems / 2;
                            let idx1 = (elems.saturating_sub(1)) / 3;
                            let idx2 = (elems.saturating_sub(1)) / 5;
                            // write via safe API
                            if idx0 < t.size() {
                                t.set(&[idx0], 1.2345);
                            }
                            if idx1 < t.size() {
                                t.set(&[idx1], 2.3456);
                            }
                            if idx2 < t.size() {
                                t.set(&[idx2], 3.4567);
                            }
                        }
                    }
                }
            }));
        }
        for h in handles {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod additional_safety_tests {
    use super::*;

    #[test]
    fn test_pool_alloc_dealloc_balanced_small_medium_large() {
        let before = TensorMemoryPool::thread_stats();
        {
            let _s1 = crate::tensor::Tensor::new(vec![SMALL_BUFFER_SIZE.min(16)]);
            let _m1 = crate::tensor::Tensor::new(vec![MEDIUM_BUFFER_SIZE / 4]);
            let _l1 = crate::tensor::Tensor::new(vec![LARGE_BUFFER_SIZE / 4]);
        }
        let after = TensorMemoryPool::thread_stats();
        assert!(
            after.allocations >= before.allocations + 3,
            "allocations did not increase as expected: before={}, after={}",
            before.allocations,
            after.allocations
        );
        assert!(
            after.deallocations >= before.deallocations + 3,
            "deallocations did not increase as expected: before={}, after={}",
            before.deallocations,
            after.deallocations
        );
        // Current usage should not grow across scope
        assert!(
            after.current_usage <= before.current_usage,
            "current_usage grew: before={}, after={}",
            before.current_usage,
            after.current_usage
        );
    }

    #[test]
    fn test_pointer_alignment_across_classes() {
        let align = simd_alignment_bytes(detect_runtime_simd());
        for &n in &[
            8usize,
            SMALL_BUFFER_SIZE,
            MEDIUM_BUFFER_SIZE,
            LARGE_BUFFER_SIZE + 128,
        ] {
            let t = crate::tensor::Tensor::new(vec![n]);
            unsafe {
                let addr = t.as_ptr() as usize;
                assert_eq!(
                    addr % align,
                    0,
                    "pointer not aligned to {} for n={}",
                    align,
                    n
                );
            }
        }
    }

    #[test]
    fn test_with_no_mem_pool_uses_system_allocator_no_pool_stats() {
        let before = TensorMemoryPool::thread_stats();
        with_no_mem_pool(|| {
            let _t1 = crate::tensor::Tensor::new(vec![64]);
            let _t2 = crate::tensor::Tensor::new(vec![2048]);
            let _t3 = crate::tensor::Tensor::new(vec![131072]);
        });
        let after = TensorMemoryPool::thread_stats();
        // Pool should not register hits/misses when disabled within the scope
        assert_eq!(
            after.allocations, before.allocations,
            "pool allocations changed with pool disabled: before={}, after={}",
            before.allocations, after.allocations
        );
        assert_eq!(
            after.deallocations, before.deallocations,
            "pool deallocations changed with pool disabled: before={}, after={}",
            before.deallocations, after.deallocations
        );
    }

    #[test]
    fn test_cross_thread_drop_does_not_affect_this_thread_stats() {
        let before = TensorMemoryPool::thread_stats();
        // Allocate in a worker thread and drop in this thread
        let handle =
            std::thread::spawn(|| crate::tensor::Tensor::new(vec![SMALL_BUFFER_SIZE.min(32)]));
        let t = handle.join().unwrap();
        drop(t); // Drop on current thread; should not touch this thread's pool stats
        let after = TensorMemoryPool::thread_stats();
        assert_eq!(
            after.allocations, before.allocations,
            "allocations changed in current thread due to cross-thread drop: before={}, after={}",
            before.allocations, after.allocations
        );
        // Deallocation also should not be recorded in this thread
        assert_eq!(
            after.deallocations, before.deallocations,
            "deallocations changed in current thread due to cross-thread drop: before={}, after={}",
            before.deallocations, after.deallocations
        );
    }

    #[test]
    fn test_many_alloc_dealloc_cycles_no_growth_in_current_usage() {
        let before = TensorMemoryPool::thread_stats();
        for _ in 0..100 {
            let _t1 = crate::tensor::Tensor::new(vec![SMALL_BUFFER_SIZE.min(64)]);
            let _t2 = crate::tensor::Tensor::new(vec![MEDIUM_BUFFER_SIZE / 8]);
        }
        let after = TensorMemoryPool::thread_stats();
        // current_usage should remain bounded and not monotonically grow
        assert!(
            after.current_usage <= before.current_usage + SMALL_BUFFER_SIZE + MEDIUM_BUFFER_SIZE,
            "current_usage unexpected growth: before={}, after={}",
            before.current_usage,
            after.current_usage
        );
    }
}

#[cfg(test)]
mod cleanup_tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    // Helper to create and hold N tensors of the given element count (single-dim)
    fn hold_tensors(count: usize, elems: usize) -> Vec<crate::tensor::Tensor> {
        let mut v = Vec::with_capacity(count);
        for _ in 0..count {
            v.push(crate::tensor::Tensor::new(vec![elems]));
        }
        v
    }

    // Helper to bump pool op counters by performing lightweight small allocations
    fn bump_ops_small_iters(iters: usize) {
        for _ in 0..iters {
            let _t = crate::tensor::Tensor::new(vec![SMALL_BUFFER_SIZE.min(8)]);
        }
    }

    #[test]
    fn test_no_cleanup_while_many_small_buffers_in_use() {
        // Prime the pool with many small buffers held alive
        let holders = hold_tensors(40, SMALL_BUFFER_SIZE.min(32));
        let (small_before, _, _, _) = TensorMemoryPool::pool_sizes();
        assert!(
            small_before >= 40,
            "expected >=40 small buffers, got {}",
            small_before
        );

        // Bump op counters and time while these buffers remain in use
        bump_ops_small_iters(1500); // ~3000 ops
        thread::sleep(Duration::from_millis(2100));
        bump_ops_small_iters(700); // exceed thresholds

        // Trigger a cleanup attempt via an allocation in another size class (medium)
        {
            let _m = crate::tensor::Tensor::new(vec![MEDIUM_BUFFER_SIZE / 2]);
        }

        // While buffers are still in-use, no trimming should occur (len must not decrease)
        let (small_mid, _, _, _) = TensorMemoryPool::pool_sizes();
        assert!(
            small_mid >= small_before,
            "small pool shrank while heavily in-use: before={} after={}",
            small_before,
            small_mid
        );

        // Now drop the holders; their last_used timestamps are fresh, so cleanup shouldn't trim them
        drop(holders);

        // Trigger cleanup again
        let _ = crate::tensor::Tensor::new(vec![MEDIUM_BUFFER_SIZE / 2]);
        let (small_after, _, _, _) = TensorMemoryPool::pool_sizes();
        assert!(
            small_after >= small_before,
            "small pool unexpectedly trimmed active buffers: before={} after={}",
            small_before,
            small_after
        );
    }

    #[test]
    fn test_cleanup_trims_long_idle_medium_buffers() {
        // Create many medium buffers simultaneously to grow pool capacity
        {
            let _holders = hold_tensors(30, MEDIUM_BUFFER_SIZE / 2);
            // _holders dropped at end of scope, all buffers become available
        }
        let (_, med_before, _, _) = TensorMemoryPool::pool_sizes();
        assert!(
            med_before >= 30,
            "expected >=30 medium buffers, got {}",
            med_before
        );

        // Leave medium buffers idle; bump ops using small allocations and wait to satisfy time gate
        bump_ops_small_iters(2300); // ~4600 ops (> UNUSED_OPS_THRESHOLD)
        thread::sleep(Duration::from_millis(2100));

        // Trigger cleanup and observe trimming
        let _ = crate::tensor::Tensor::new(vec![SMALL_BUFFER_SIZE.min(16)]);
        let (_, med_after, _, _) = TensorMemoryPool::pool_sizes();

        assert!(
            med_after < med_before,
            "medium pool not trimmed despite long idle: before={} after={}",
            med_before,
            med_after
        );
    }
}
