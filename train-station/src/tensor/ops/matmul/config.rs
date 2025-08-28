//! Matrix multiplication configuration and kernel selection
//!
//! This module provides dynamic configuration for matrix multiplication operations
//! based on matrix dimensions and hardware characteristics. The configuration system
//! selects optimal kernel types and block sizes for different matrix sizes commonly
//! found in machine learning workloads.
//!
//! # Key Features
//!
//! - **Dynamic Configuration**: Runtime kernel selection based on matrix dimensions
//! - **Cache Optimization**: Block sizes optimized for modern CPU cache hierarchies
//! - **ML Workload Tuning**: Optimized for common neural network matrix dimensions
//! - **Performance Adaptation**: Automatic selection of optimal kernels and block sizes
//! - **Minimal Overhead**: Runtime configuration with compile-time optimization
//!
//! # Design Philosophy
//!
//! The configuration system prioritizes maintainability and performance by using
//! simplified adaptive blocking that automatically selects appropriate kernels
//! based on matrix dimensions. This approach balances code complexity with
//! performance optimization for common ML matrix sizes.
//!
//! # Kernel Selection Strategy
//!
//! - **Small matrices (<64 elements)**: Direct computation with minimal overhead
//! - **Larger matrices (64+ elements)**: Balanced approach with cache-aware blocking
//! - **Block size optimization**: Adapts to L1/L2 cache sizes for memory efficiency
//!
//! # Performance Characteristics
//!
//! - **Cache-friendly blocking**: Block sizes optimized for modern CPU cache hierarchies
//! - **Minimal configuration overhead**: Runtime configuration with compile-time optimization
//! - **ML workload optimization**: Tuned for common neural network matrix dimensions
//! - **Memory bandwidth optimization**: Block sizes adapted for maximum memory throughput
//!
//! # Examples
//!
//! ## Implementation Details
//!
//! The configuration system automatically selects optimal kernels and block sizes:
//!
//! - **Small matrices (<64 elements)**: Use `Direct2x8` kernel with minimal blocking
//! - **Medium matrices (64-256 elements)**: Use `Balanced4x8` kernel with L1 cache optimization
//! - **Large matrices (256+ elements)**: Use `Balanced4x8` kernel with memory bandwidth optimization
//!
//! Block sizes are dynamically adjusted based on matrix dimensions:
//!
//! - **Very small (≤32)**: 32x32x32 blocks for minimal overhead
//! - **Small (≤64)**: 64x64x64 blocks optimized for L1 cache
//! - **Medium (≤128)**: 64x128x128 blocks optimized for L2 cache
//! - **Large (>128)**: 128x128x128 blocks optimized for memory bandwidth

/// Kernel types for different matrix multiplication scenarios
///
/// Defines the available kernel implementations optimized for different matrix
/// characteristics and sizes. Each kernel type is tuned for specific performance
/// characteristics and memory access patterns.
///
/// # Variants
///
/// * `Direct2x8` - Optimized for small matrices with minimal overhead and direct computation
/// * `Balanced4x8` - General purpose kernel for larger matrices with cache-aware blocking
///
/// # Performance Characteristics
///
/// - **Direct2x8**: Minimal setup overhead, optimized for matrices <64 elements
/// - **Balanced4x8**: Cache-friendly blocking, optimized for matrices 64+ elements
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelType {
    Direct2x8,   // Small matrices, minimal overhead
    Balanced4x8, // General purpose, square matrices
}

/// Matrix multiplication configuration determined at runtime
///
/// Contains the optimal configuration parameters for matrix multiplication based on
/// matrix dimensions and hardware characteristics. The configuration is determined
/// dynamically to balance performance and memory efficiency.
///
/// # Fields
///
/// * `block_m` - Block size for the M dimension (rows of first matrix)
/// * `block_n` - Block size for the N dimension (columns of second matrix)
/// * `block_k` - Block size for the K dimension (inner dimension)
/// * `kernel_type` - Selected kernel implementation for optimal performance
///
/// # Performance Characteristics
///
/// Block sizes are optimized for cache efficiency and memory bandwidth utilization.
/// The configuration adapts to matrix dimensions to maximize performance for
/// common machine learning workload sizes.
#[derive(Debug, Clone)]
pub struct MatmulConfig {
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub _kernel_type: KernelType,
}

impl MatmulConfig {
    /// Determine optimal configuration based on matrix dimensions
    ///
    /// Analyzes matrix dimensions to select the most efficient kernel type and block sizes
    /// for the given matrix multiplication. The selection prioritizes performance for
    /// common machine learning workload sizes while maintaining code simplicity.
    ///
    /// # Arguments
    ///
    /// * `m` - Number of rows in the first matrix (result rows)
    /// * `n` - Number of columns in the second matrix (result columns)
    /// * `k` - Inner dimension (columns of first matrix, rows of second matrix)
    ///
    /// # Returns
    ///
    /// Optimal configuration with kernel type and block sizes for the given dimensions
    ///
    /// # Implementation Details
    ///
    /// - Small matrices (<64 elements): Use lightweight kernels with minimal overhead
    /// - Larger matrices: Use balanced approach with cache-aware blocking
    /// - Block sizes adapt to L1/L2 cache sizes for memory efficiency
    pub fn for_dimensions(m: usize, n: usize, k: usize) -> Self {
        // Analyze matrix characteristics for kernel selection
        let max_dim = m.max(n).max(k);

        // Select optimal kernel type based on matrix size
        let kernel_type = if max_dim < 64 {
            // Small matrices: Use lightweight kernels
            KernelType::Direct2x8
        } else {
            // Larger matrices: Use balanced approach
            KernelType::Balanced4x8
        };

        // Simplified block sizes for cache efficiency
        let (block_m, block_n, block_k) = if max_dim <= 32 {
            // Very small matrices: minimal blocking
            (32.min(m), 32.min(n), 32.min(k))
        } else if max_dim <= 64 {
            // Small matrices: optimized for L1 cache
            (64.min(m), 64.min(n), 64.min(k))
        } else if max_dim <= 128 {
            // Medium matrices: optimized for L2 cache
            (64.min(m), 128.min(n), 128.min(k))
        } else {
            // Large matrices: memory bandwidth optimized
            (128.min(m), 128.min(n), 128.min(k))
        };

        Self {
            block_m,
            block_n,
            block_k,
            _kernel_type: kernel_type,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Matrix multiplication configuration tests
    //!
    //! Tests for kernel selection logic and configuration generation based on
    //! matrix dimensions. Validates that appropriate kernel types and block
    //! sizes are selected for different matrix sizes.

    use super::*;

    /// Test configuration for small matrix dimensions
    ///
    /// Verifies that small matrices (<64 elements) are assigned the Direct2x8 kernel
    /// type with appropriate block sizes for minimal overhead computation.
    #[test]
    fn test_config_small_matrices() {
        let config = MatmulConfig::for_dimensions(16, 16, 16);
        assert_eq!(config._kernel_type, KernelType::Direct2x8);
        assert!(config.block_m <= 32);
        assert!(config.block_n <= 32);
        assert!(config.block_k <= 32);
    }

    /// Test configuration for medium matrix dimensions
    ///
    /// Verifies that medium-sized matrices (64-128 elements) are assigned the
    /// Balanced4x8 kernel type with cache-optimized block sizes.
    #[test]
    fn test_config_medium_matrices() {
        let config = MatmulConfig::for_dimensions(64, 64, 64);
        assert_eq!(config._kernel_type, KernelType::Balanced4x8);
        assert!(config.block_m <= 64);
        assert!(config.block_n <= 128);
        assert!(config.block_k <= 128);
    }

    /// Test configuration for large matrix dimensions
    ///
    /// Verifies that large matrices (256+ elements) are assigned the Balanced4x8
    /// kernel type with memory bandwidth optimized block sizes.
    #[test]
    fn test_config_large_matrices() {
        let config = MatmulConfig::for_dimensions(512, 512, 512);
        assert_eq!(config._kernel_type, KernelType::Balanced4x8);
        assert!(config.block_m <= 128);
        assert!(config.block_n <= 128);
        assert!(config.block_k <= 128);
    }
}
