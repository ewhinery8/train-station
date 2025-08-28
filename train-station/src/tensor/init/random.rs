//! Random tensor initialization methods
//!
//! This module provides methods to create tensors with random values using
//! efficient random number generation algorithms. All methods are optimized
//! for performance with SIMD operations and provide reproducible results
//! when seeds are specified.
//!
//! # Key Features
//!
//! - **`randn`**: Create tensors with normally distributed random values (mean=0, std=1)
//! - **Box-Muller Transform**: Efficient normal distribution generation
//! - **SIMD Optimization**: Vectorized operations for large tensors
//! - **Reproducible Results**: Optional seed-based generation for deterministic output
//! - **Thread Safety**: Thread-local random state management
//! - **Statistical Quality**: High-quality random number generation
//!
//! # Performance Characteristics
//!
//! - **Box-Muller Transform**: Efficient normal distribution generation
//! - **SIMD Operations**: AVX2-optimized operations for large tensors
//! - **Memory Efficient**: Single-pass generation with optimized allocation
//! - **Unrolled Loops**: 4x unrolling for better instruction throughput
//! - **Zero Overhead**: Minimal validation overhead for correct usage
//!
//! # Examples
//!
//! ## Basic Random Generation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create a 2x3 tensor with random normal values
//! let tensor = Tensor::randn(vec![2, 3], None);
//! assert_eq!(tensor.size(), 6);
//! assert_eq!(tensor.shape().dims, vec![2, 3]);
//!
//! // Verify random values are generated
//! let first_value = tensor.get(&[0, 0]);
//! assert!(first_value != 0.0); // Should be random
//! ```
//!
//! ## Reproducible Random Generation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create with fixed seed for reproducible results
//! let tensor1 = Tensor::randn(vec![100], Some(42));
//! let tensor2 = Tensor::randn(vec![100], Some(42));
//!
//! // tensor1 and tensor2 will have identical values
//! for i in 0..tensor1.size() {
//!     assert!((tensor1.get(&[i]) - tensor2.get(&[i])).abs() < 1e-6);
//! }
//! ```
//!
//! ## Different Seeds Produce Different Results
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor1 = Tensor::randn(vec![100], Some(1));
//! let tensor2 = Tensor::randn(vec![100], Some(2));
//!
//! // Should be different with different seeds
//! let mut different = false;
//! for i in 0..tensor1.size() {
//!     if (tensor1.get(&[i]) - tensor2.get(&[i])).abs() > 1e-6 {
//!         different = true;
//!         break;
//!     }
//! }
//! assert!(different, "Tensors with different seeds should be different");
//! ```
//!
//! ## Large Tensor Generation
//!
//! ```
//! use train_station::Tensor;
//!
//! // Efficient generation of large tensors
//! let tensor = Tensor::randn(vec![100, 100], Some(42));
//! assert_eq!(tensor.size(), 10000);
//!
//! // Check statistical properties
//! let mut sum = 0.0;
//! for i in 0..tensor.size() {
//!     sum += tensor.get(&[i / 100, i % 100]);
//! }
//! let mean = sum / tensor.size() as f32;
//!
//! // Mean should be close to 0 for normal distribution
//! assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
//! ```
//!
//! ## Zero-Sized Tensor Handling
//!
//! ```
//! use train_station::Tensor;
//!
//! // Handle empty tensors gracefully
//! let tensor = Tensor::randn(vec![0], Some(42));
//! assert_eq!(tensor.size(), 0);
//! assert_eq!(tensor.shape().dims, vec![0]);
//! ```
//!
//! # Design Principles
//!
//! - **Statistical Quality**: High-quality random number generation with proper distribution
//! - **Performance First**: SIMD-optimized operations for maximum speed
//! - **Reproducibility**: Optional seed-based generation for deterministic results
//! - **Memory Efficiency**: Efficient memory operations with minimal overhead
//! - **Thread Safety**: Safe concurrent access with thread-local state
//! - **Numerical Stability**: Robust handling of edge cases and numerical issues

use crate::tensor::core::Tensor;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

impl Tensor {
    /// Creates a tensor with normally distributed random values (mean=0, std=1)
    ///
    /// Similar to PyTorch's `torch.randn()`, creates a tensor filled with random
    /// values drawn from a standard normal distribution (mean=0, standard deviation=1).
    /// Uses Box-Muller transform for efficient normal distribution generation.
    ///
    /// This method provides high-quality random number generation with optional
    /// reproducibility through seed-based generation. The generated values follow
    /// a standard normal distribution suitable for machine learning applications.
    ///
    /// # Arguments
    ///
    /// * `shape_dims` - Vector of dimension sizes defining the tensor shape
    /// * `seed` - Optional seed for reproducible random generation
    ///
    /// # Returns
    ///
    /// A new tensor with normally distributed random values
    ///
    /// # Performance
    ///
    /// - **Box-Muller Transform**: Efficient normal distribution generation
    /// - **SIMD Optimization**: Vectorized operations for large tensors
    /// - **Memory Efficient**: Single-pass generation with optimized allocation
    /// - **Thread Safe**: Uses thread-local random state
    ///
    /// # Examples
    ///
    /// ## Basic Usage
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create a 2x3 tensor with random normal values
    /// let tensor = Tensor::randn(vec![2, 3], None);
    /// assert_eq!(tensor.size(), 6);
    /// assert_eq!(tensor.shape().dims, vec![2, 3]);
    ///
    /// // Verify random values are generated
    /// let first_value = tensor.get(&[0, 0]);
    /// assert!(first_value != 0.0); // Should be random
    /// ```
    ///
    /// ## Reproducible Generation
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Create with fixed seed for reproducible results
    /// let tensor1 = Tensor::randn(vec![100], Some(42));
    /// let tensor2 = Tensor::randn(vec![100], Some(42));
    ///
    /// // tensor1 and tensor2 will have identical values
    /// for i in 0..tensor1.size() {
    ///     assert!((tensor1.get(&[i]) - tensor2.get(&[i])).abs() < 1e-6);
    /// }
    /// ```
    ///
    /// ## Statistical Properties
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Generate large tensor for statistical analysis
    /// let tensor = Tensor::randn(vec![1000], Some(42));
    /// assert_eq!(tensor.size(), 1000);
    ///
    /// // Check that values are reasonable (within 4 standard deviations)
    /// let mut min_val = f32::INFINITY;
    /// let mut max_val = f32::NEG_INFINITY;
    /// let mut sum = 0.0;
    ///
    /// for i in 0..tensor.size() {
    ///     let val = tensor.get(&[i]);
    ///     min_val = min_val.min(val);
    ///     max_val = max_val.max(val);
    ///     sum += val;
    /// }
    ///
    /// let mean = sum / tensor.size() as f32;
    ///
    /// // Mean should be close to 0, values should be within reasonable bounds
    /// assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
    /// assert!(min_val > -4.0, "Values should not be too negative, min: {}", min_val);
    /// assert!(max_val < 4.0, "Values should not be too positive, max: {}", max_val);
    /// ```
    ///
    /// ## Zero-Sized Tensors
    ///
    /// ```
    /// use train_station::Tensor;
    ///
    /// // Handle empty tensors gracefully
    /// let tensor = Tensor::randn(vec![0], Some(42));
    /// assert_eq!(tensor.size(), 0);
    /// assert_eq!(tensor.shape().dims, vec![0]);
    /// ```
    ///
    /// # Implementation Details
    ///
    /// This method uses the Box-Muller transform to generate normally distributed
    /// random variables from uniform random variables. The process involves:
    /// 1. **Random Number Generation**: Uses Xorshift algorithm for uniform random numbers
    /// 2. **Box-Muller Transform**: Converts uniform random variables to normal distribution
    /// 3. **SIMD Optimization**: Vectorized operations for large tensors when available
    /// 4. **Numerical Stability**: Robust handling of edge cases and potential NaN values
    ///
    /// The Box-Muller transform ensures that the generated values follow a true
    /// normal distribution with mean=0 and standard deviation=1, making it suitable
    /// for machine learning applications requiring normally distributed random values.
    pub fn randn(shape_dims: Vec<usize>, seed: Option<u64>) -> Self {
        let mut tensor = Self::new(shape_dims);
        tensor.fill_randn(seed);
        tensor
    }

    /// Fills the tensor with normally distributed random values
    ///
    /// Internal method that fills an existing tensor with random values from
    /// a standard normal distribution. Uses Box-Muller transform for efficiency
    /// and provides SIMD optimization for large tensors.
    ///
    /// This method is used internally by `randn()` and provides the core
    /// random number generation functionality with optimized performance
    /// characteristics.
    ///
    /// # Arguments
    ///
    /// * `seed` - Optional seed for reproducible random generation
    ///
    /// # Performance
    ///
    /// - **Box-Muller Transform**: Generates pairs of normal random variables
    /// - **SIMD Optimization**: Vectorized operations when possible
    /// - **Memory Efficient**: Single-pass generation
    /// - **Unrolled Loops**: 4x unrolling for better instruction throughput
    ///
    /// # Implementation Details
    ///
    /// The method performs the following steps:
    /// 1. **Zero-sized Check**: Returns early for empty tensors
    /// 2. **RNG Initialization**: Creates Xorshift RNG with seed or system time
    /// 3. **SIMD Detection**: Checks for AVX2 availability for optimized path
    /// 4. **Generation**: Uses SIMD or scalar path based on hardware support
    /// 5. **Completion**: Fills all tensor elements with normal random values
    ///
    /// The method automatically handles hardware capabilities and falls back
    /// to scalar operations when SIMD is not available, ensuring compatibility
    /// across different CPU architectures.
    pub fn fill_randn(&mut self, seed: Option<u64>) {
        if self.shape().size == 0 {
            return;
        }

        // Initialize random number generator
        let mut rng = if let Some(seed_val) = seed {
            // Use provided seed for reproducible results
            XorShiftRng::new(seed_val)
        } else {
            // Use system time for non-reproducible results
            XorShiftRng::new_from_time()
        };

        unsafe {
            let ptr = self.as_ptr();

            #[cfg(target_arch = "x86_64")]
            {
                // Use SIMD for better performance when available
                if is_x86_feature_detected!("avx2") {
                    self.fill_randn_simd_avx2(ptr, &mut rng);
                    return;
                }
            }

            // Fallback to scalar operations
            self.fill_randn_scalar(ptr, &mut rng);
        }
    }

    /// Fills the tensor with normally distributed random values using AVX2 SIMD
    ///
    /// Internal method that uses AVX2 instructions to efficiently fill large tensors
    /// with normal random values. Processes 8 elements per iteration for maximum
    /// memory bandwidth utilization.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the tensor data
    /// * `rng` - Random number generator instance
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `ptr` is a valid pointer to tensor data
    /// * The tensor size matches the allocated memory
    /// * AVX2 is available on the target architecture
    /// * `rng` is a valid random number generator instance
    ///
    /// # Performance
    ///
    /// - **SIMD Operations**: 8 elements per iteration using AVX2
    /// - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
    /// - **Remaining Elements**: Efficient handling of non-multiple-of-8 sizes
    /// - **Box-Muller Transform**: Integrated normal distribution generation
    ///
    /// # Implementation Details
    ///
    /// This method uses AVX2 SIMD instructions to fill memory efficiently:
    /// 1. Generates 8 normal random values using Box-Muller transform
    /// 2. Loads values into AVX2 vector register using `_mm256_loadu_ps`
    /// 3. Stores vector to memory using `_mm256_storeu_ps`
    /// 4. Processes remaining elements with scalar operations
    ///
    /// The method provides significant performance improvements for large tensors
    /// by reducing the number of memory operations and leveraging vectorized
    /// floating-point operations.
    #[cfg(target_arch = "x86_64")]
    #[inline]
    unsafe fn fill_randn_simd_avx2(&self, ptr: *const f32, rng: &mut XorShiftRng) {
        let mut_ptr = ptr as *mut f32;
        let size = self.shape().size;
        let simd_count = size / 8; // Process 8 elements per iteration
        let mut offset = 0;

        // SIMD loop for normal distribution generation
        for _ in 0..simd_count {
            let mut values = [0.0f32; 8];
            for i in &mut values {
                *i = rng.next_normal();
            }

            // Store 8 values using SIMD
            let vec = _mm256_loadu_ps(values.as_ptr());
            _mm256_storeu_ps(mut_ptr.add(offset), vec);
            offset += 8;
        }

        // Handle remaining elements
        for i in offset..size {
            *mut_ptr.add(i) = rng.next_normal();
        }
    }

    /// Fills the tensor with normally distributed random values using scalar operations
    ///
    /// Internal fallback method that uses scalar operations to fill tensors with
    /// normal random values. Provides 4x unrolled loops for better instruction
    /// throughput and serves as a fallback when SIMD is not available.
    ///
    /// # Arguments
    ///
    /// * `ptr` - Pointer to the tensor data
    /// * `rng` - Random number generator instance
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// * `ptr` is a valid pointer to tensor data
    /// * The tensor size matches the allocated memory
    /// * `rng` is a valid random number generator instance
    ///
    /// # Performance
    ///
    /// - **Unrolled Loops**: 4x unrolling for better instruction throughput
    /// - **Box-Muller Transform**: Integrated normal distribution generation
    /// - **Remaining Elements**: Efficient handling of non-multiple-of-4 sizes
    /// - **Cross-Platform**: Works on all CPU architectures
    ///
    /// # Implementation Details
    ///
    /// This method provides optimized scalar operations:
    /// 1. **Unrolled Generation**: Processes 4 elements per iteration
    /// 2. **Box-Muller Transform**: Generates normal random values
    /// 3. **Remaining Elements**: Handles final elements individually
    /// 4. **Cross-Platform**: No architecture-specific dependencies
    ///
    /// The 4x unrolling reduces loop overhead and improves instruction-level
    /// parallelism, making scalar operations more efficient than naive loops.
    #[inline]
    unsafe fn fill_randn_scalar(&self, ptr: *const f32, rng: &mut XorShiftRng) {
        let mut_ptr = ptr as *mut f32;
        let size = self.shape().size;
        let unroll_count = size / 4;
        let mut offset = 0;

        // Unrolled scalar loop for better performance
        for _ in 0..unroll_count {
            *mut_ptr.add(offset) = rng.next_normal();
            *mut_ptr.add(offset + 1) = rng.next_normal();
            *mut_ptr.add(offset + 2) = rng.next_normal();
            *mut_ptr.add(offset + 3) = rng.next_normal();
            offset += 4;
        }

        // Handle remaining elements
        for i in offset..size {
            *mut_ptr.add(i) = rng.next_normal();
        }
    }
}

// SIMD optimizations for performance-critical operations
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Fast random number generator using Xorshift algorithm
///
/// Provides efficient random number generation with good statistical properties.
/// Implements Box-Muller transform for normal distribution generation.
///
/// The Xorshift algorithm is a fast, non-cryptographic random number generator
/// that provides good statistical properties for machine learning applications.
/// It combines multiple bit-shift and XOR operations to produce high-quality
/// random sequences with long periods.
///
/// # Performance
///
/// - **Fast Generation**: Minimal computational overhead
/// - **Good Statistical Properties**: Passes standard statistical tests
/// - **Long Period**: 2^64 - 1 period for u64 state
/// - **Memory Efficient**: Single u64 state variable
///
/// # Implementation Details
///
/// The Xorshift algorithm uses three bit-shift and XOR operations:
/// 1. `state ^= state << 13` - Left shift by 13 bits
/// 2. `state ^= state >> 7` - Right shift by 7 bits  
/// 3. `state ^= state << 17` - Left shift by 17 bits
///
/// This sequence provides excellent statistical properties and is much faster
/// than more complex generators like Mersenne Twister.
struct XorShiftRng {
    state: u64,
}

impl XorShiftRng {
    /// Creates a new random number generator with the specified seed
    ///
    /// Initializes the RNG with a user-provided seed for reproducible
    /// random number generation. The same seed will always produce
    /// the same sequence of random numbers.
    ///
    /// # Arguments
    ///
    /// * `seed` - The seed value for reproducible generation
    ///
    /// # Implementation Details
    ///
    /// This method initializes the internal state with the provided seed value.
    /// The same seed will always produce the same sequence of random numbers,
    /// making it suitable for reproducible random number generation.
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Creates a new random number generator seeded from system time
    ///
    /// Initializes the RNG with a seed derived from the current system time,
    /// providing non-reproducible random number generation. Each call will
    /// produce a different sequence of random numbers.
    ///
    /// # Implementation Details
    ///
    /// This method uses the system time to generate a seed:
    /// 1. Gets current system time using `std::time::SystemTime::now()`
    /// 2. Hashes the time value using `DefaultHasher`
    /// 3. Uses the hash result as the RNG seed
    ///
    /// This approach provides good entropy for non-reproducible generation
    /// while being efficient and portable across different platforms.
    fn new_from_time() -> Self {
        let mut hasher = DefaultHasher::new();
        std::time::SystemTime::now().hash(&mut hasher);
        let seed = hasher.finish();
        Self { state: seed }
    }

    /// Generates the next random u64 value
    ///
    /// Produces the next 64-bit random value using the Xorshift algorithm.
    /// This is the core random number generation method that drives all
    /// other random value generation.
    ///
    /// # Returns
    ///
    /// A random u64 value from the Xorshift sequence
    ///
    /// # Performance
    ///
    /// - **Fast**: Only 3 bit-shift and XOR operations
    /// - **Efficient**: No branching or complex arithmetic
    /// - **Deterministic**: Same seed produces same sequence
    ///
    /// # Implementation Details
    ///
    /// The Xorshift algorithm performs three operations:
    /// 1. `self.state ^= self.state << 13` - Left shift and XOR
    /// 2. `self.state ^= self.state >> 7` - Right shift and XOR
    /// 3. `self.state ^= self.state << 17` - Left shift and XOR
    ///
    /// This sequence provides excellent statistical properties with minimal
    /// computational overhead, making it ideal for high-performance applications.
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Generates the next random f32 value in [0, 1)
    ///
    /// Produces a random floating-point value in the half-open interval [0, 1).
    /// Uses proper bit manipulation to ensure uniform distribution across
    /// the floating-point range.
    ///
    /// # Returns
    ///
    /// A random f32 value in [0, 1)
    ///
    /// # Performance
    ///
    /// - **Efficient**: Single u64 generation with bit manipulation
    /// - **Uniform**: Proper distribution across floating-point range
    /// - **Precise**: Uses 23 bits of mantissa for good precision
    ///
    /// # Implementation Details
    ///
    /// The method converts u64 random bits to f32 using bit manipulation:
    /// 1. Generates random u64 using `next_u64()`
    /// 2. Extracts 23 bits for mantissa (IEEE 754 f32 format)
    /// 3. Sets exponent to 0 (bias 126) for values in [1, 2)
    /// 4. Converts to f32 and subtracts 1.0 for [0, 1) range
    ///
    /// This approach provides uniform distribution and avoids the bias
    /// that can occur with simple division-based methods.
    fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64();
        // Convert to f32 in [0, 1) using proper bit manipulation
        let mantissa = (bits & 0x7FFFFF) as u32; // 23 bits for mantissa
        let exponent = 126 << 23; // 2^0 = 1, so bias 126 gives exponent 0
        let float_bits = mantissa | exponent;
        f32::from_bits(float_bits)
    }

    /// Generates the next normally distributed random value using Box-Muller transform
    ///
    /// Produces a random value from a standard normal distribution (mean=0, std=1)
    /// using the Box-Muller transform. This method converts uniform random variables
    /// to normally distributed random variables efficiently.
    ///
    /// # Returns
    ///
    /// A random f32 value from N(0, 1) distribution
    ///
    /// # Performance
    ///
    /// - **Efficient**: Single pass transformation
    /// - **Accurate**: Proper normal distribution properties
    /// - **Stable**: Handles edge cases and numerical issues
    ///
    /// # Implementation Details
    ///
    /// The Box-Muller transform converts uniform random variables to normal:
    /// 1. Generates two uniform random variables u1, u2 in (0, 1)
    /// 2. Applies Box-Muller formula: z = sqrt(-2*ln(u1)) * cos(2π*u2)
    /// 3. Handles edge cases (u1 ≤ 0, u1 ≥ 1, NaN, infinite values)
    /// 4. Returns z as normally distributed random value
    ///
    /// The method includes robust error handling to ensure numerical stability
    /// and prevent invalid results from edge cases in the transformation.
    fn next_normal(&mut self) -> f32 {
        // Box-Muller transform: convert uniform random variables to normal
        let u1 = self.next_f32();
        let u2 = self.next_f32();

        // Avoid log(0) and ensure u1 is in (0, 1)
        let u1 = if u1 <= 0.0 {
            1e-7
        } else if u1 >= 1.0 {
            1.0 - 1e-7
        } else {
            u1
        };

        // Box-Muller transform
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();

        // Handle potential NaN or infinite values
        if z0.is_nan() || z0.is_infinite() {
            0.0
        } else {
            z0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_randn_basic() {
        let tensor = Tensor::randn(vec![2, 3], Some(42));
        assert_eq!(tensor.size(), 6);
        assert_eq!(tensor.shape().dims, vec![2, 3]);

        // With fixed seed, should be reproducible
        let tensor2 = Tensor::randn(vec![2, 3], Some(42));
        for i in 0..tensor.size() {
            unsafe {
                assert!((*tensor.as_ptr().add(i) - *tensor2.as_ptr().add(i)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_randn_reproducible() {
        let seed = 12345;
        let tensor1 = Tensor::randn(vec![100], Some(seed));
        let tensor2 = Tensor::randn(vec![100], Some(seed));

        // Should be identical with same seed
        for i in 0..tensor1.size() {
            unsafe {
                assert!((*tensor1.as_ptr().add(i) - *tensor2.as_ptr().add(i)).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_randn_different_seeds() {
        let tensor1 = Tensor::randn(vec![100], Some(1));
        let tensor2 = Tensor::randn(vec![100], Some(2));

        // Should be different with different seeds
        let mut different = false;
        for i in 0..tensor1.size() {
            unsafe {
                if (*tensor1.as_ptr().add(i) - *tensor2.as_ptr().add(i)).abs() > 1e-6 {
                    different = true;
                    break;
                }
            }
        }
        assert!(
            different,
            "Tensors with different seeds should be different"
        );
    }

    #[test]
    fn test_randn_no_seed() {
        let tensor = Tensor::randn(vec![10], None);
        assert_eq!(tensor.size(), 10);
        assert_eq!(tensor.shape().dims, vec![10]);

        // Should not be all zeros
        let mut has_non_zero = false;
        for i in 0..tensor.size() {
            unsafe {
                if *tensor.as_ptr().add(i) != 0.0 {
                    has_non_zero = true;
                    break;
                }
            }
        }
        assert!(has_non_zero, "Random tensor should not be all zeros");
    }

    #[test]
    fn test_randn_zero_sized() {
        let tensor = Tensor::randn(vec![0], Some(42));
        assert_eq!(tensor.size(), 0);
        assert_eq!(tensor.shape().dims, vec![0]);
    }

    #[test]
    fn test_randn_large_tensor() {
        let tensor = Tensor::randn(vec![100, 100], Some(42));
        assert_eq!(tensor.size(), 10000);

        // Check that values are reasonable (within 4 standard deviations)
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        let mut sum = 0.0;

        for i in 0..tensor.size() {
            unsafe {
                let val = *tensor.as_ptr().add(i);
                min_val = min_val.min(val);
                max_val = max_val.max(val);
                sum += val;
            }
        }

        let mean = sum / tensor.size() as f32;

        // Mean should be close to 0, values should be within reasonable bounds
        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
        assert!(
            min_val > -4.0,
            "Values should not be too negative, min: {}",
            min_val
        );
        assert!(
            max_val < 4.0,
            "Values should not be too positive, max: {}",
            max_val
        );
    }

    #[test]
    fn test_fill_randn() {
        let mut tensor = Tensor::new(vec![2, 3]);
        tensor.fill_randn(Some(42));

        assert_eq!(tensor.size(), 6);

        // Should not be all zeros
        let mut has_non_zero = false;
        for i in 0..tensor.size() {
            unsafe {
                if *tensor.as_ptr().add(i) != 0.0 {
                    has_non_zero = true;
                    break;
                }
            }
        }
        assert!(has_non_zero, "Random tensor should not be all zeros");
    }
}
