//! Tensor operation implementations
//!
//! This module contains all tensor operations organized by functionality for maximum
//! performance and maintainability. Each operation includes optimized SIMD implementations,
//! comprehensive testing, and automatic differentiation support.
//!
//! # Key Features
//!
//! - **SIMD Optimization**: AVX2 implementations for x86_64 architectures
//! - **Numerical Stability**: Careful implementation of mathematical functions
//! - **GradTrack Integration**: Automatic gradient computation for all operations
//! - **Performance Tuning**: Size-specific optimizations and cache-aware algorithms
//! - **Memory Safety**: Bounds checking and proper memory management
//! - **Thread Safety**: All operations are thread-safe and Send + Sync
//! - **Operator Overloading**: Natural mathematical expressions through traits
//! - **Broadcasting Support**: Automatic shape broadcasting for element-wise operations
//! - **Zero-Cost Abstractions**: Minimal overhead for operation dispatch
//!
//! # Performance Characteristics
//!
//! - **SIMD Operations**: 8x vectorization for element-wise operations
//! - **Cache Optimization**: Blocked algorithms for large matrix operations
//! - **Memory Access**: Optimized patterns for CPU cache hierarchies
//! - **Scalar Fallbacks**: Efficient fallbacks for non-SIMD hardware
//! - **Zero-Cost Dispatch**: Compile-time operation selection
//! - **Minimal Allocations**: Reuse of temporary buffers where possible
//! - **Memory Bandwidth**: Optimized for maximum memory bandwidth utilization
//!
//! # Organization
//!
//! The operations module is organized into specialized files by operation type:
//! - **Element-wise Operations**: `add`, `sub`, `mul`, `div` with SIMD optimization
//! - **Activation Functions**: `relu`, `leaky_relu`, `sigmoid`, `tanh`, `softmax`
//! - **Mathematical Functions**: `exp`, `log`, `sqrt`, `pow` with numerical stability
//! - **Matrix Operations**: `matmul` with blocked algorithms and kernel optimization
//! - **Broadcasting**: Automatic shape broadcasting for element-wise operations
//!
//! # Examples
//!
//! ## Element-wise Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! // Create tensors for operations
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // Element-wise operations
//! let result = a.add_tensor(&b);         // Tensor addition
//! assert_eq!(result.data(), &[6.0, 8.0, 10.0, 12.0]);
//!
//! let result = a.sub_tensor(&b);         // Tensor subtraction
//! assert_eq!(result.data(), &[-4.0, -4.0, -4.0, -4.0]);
//!
//! let result = a.mul_tensor(&b);         // Element-wise multiplication
//! assert_eq!(result.data(), &[5.0, 12.0, 21.0, 32.0]);
//!
//! let result = a.div_tensor(&b);         // Element-wise division
//! assert_eq!(result.data(), &[0.2, 0.33333334, 0.42857143, 0.5]);
//! ```
//!
//! ## Scalar Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Scalar operations
//! let result = tensor.add_scalar(5.0);        // Tensor + scalar
//! assert_eq!(result.data(), &[6.0, 7.0, 8.0, 9.0]);
//!
//! let result = tensor.mul_scalar(3.0);        // Tensor * scalar
//! assert_eq!(result.data(), &[3.0, 6.0, 9.0, 12.0]);
//! ```
//!
//! ## Activation Functions
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();
//!
//! // Activation functions
//! let result = tensor.relu();                 // ReLU activation
//! assert_eq!(result.data(), &[0.0, 0.0, 1.0, 2.0]);
//!
//! let result = tensor.sigmoid();              // Sigmoid activation
//! assert!(result.data()[0] < 0.5); // Negative values become < 0.5
//! assert!(result.data()[3] > 0.5); // Positive values become > 0.5
//!
//! let result = tensor.tanh();                 // Hyperbolic tangent
//! assert!(result.data()[0] < 0.0); // Negative values become negative
//! assert!(result.data()[3] > 0.0); // Positive values become positive
//! ```
//!
//! ## Mathematical Functions
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Mathematical functions
//! let result = tensor.exp();                  // Exponential
//! assert!(result.data()[0] > 2.0); // e^1 > 2
//!
//! let result = tensor.sqrt();                 // Square root
//! assert_eq!(result.data()[0], 1.0); // sqrt(1) = 1
//! assert_eq!(result.data()[3], 2.0); // sqrt(4) = 2
//!
//! let result = tensor.pow_scalar(2.0);        // Power function
//! assert_eq!(result.data()[0], 1.0); // 1^2 = 1
//! assert_eq!(result.data()[3], 16.0); // 4^2 = 16
//! ```
//!
//! ## Matrix Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // Matrix operations
//! let result = a.matmul(&b);             // Matrix multiplication
//! assert_eq!(result.shape().dims, vec![2, 2]);
//! assert_eq!(result.data()[0], 19.0); // 1*5 + 2*7 = 19
//! assert_eq!(result.data()[1], 22.0); // 1*6 + 2*8 = 22
//! ```
//!
//! ## Gradient Tracking
//!
//! ```
//! use train_station::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // With gradient tracking
//! let a_with_grad = a.with_requires_grad();
//! let result = a_with_grad.add_tensor(&b);
//!
//! // Verify the operation works correctly
//! assert_eq!(result.data(), &[6.0, 8.0, 10.0, 12.0]);
//! assert!(result.requires_grad());
//! ```
//!
//! ## Broadcasting
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let scalar = Tensor::from_slice(&[5.0], vec![1]).unwrap();
//!
//! // Broadcasting automatically handles shape differences
//! let result = tensor.add_tensor(&scalar);
//! assert_eq!(result.data(), &[6.0, 7.0, 8.0, 9.0]);
//! ```
//!
//! # Operation Categories
//!
//! ## Element-wise Operations
//! - **`add`**: Tensor and scalar addition with broadcasting
//! - **`sub`**: Tensor and scalar subtraction with broadcasting
//! - **`mul`**: Element-wise multiplication with broadcasting
//! - **`div`**: Element-wise division with broadcasting
//!
//! ## Activation Functions
//! - **`relu`**: Rectified Linear Unit with SIMD optimization
//! - **`leaky_relu`**: Leaky ReLU with configurable slope
//! - **`sigmoid`**: Sigmoid activation with numerical stability
//! - **`tanh`**: Hyperbolic tangent with accurate implementation
//! - **`softmax`**: Softmax with numerical stability and dimension support
//!
//! ## Mathematical Functions
//! - **`exp`**: Exponential function with overflow protection
//! - **`log`**: Natural logarithm with domain validation
//! - **`sqrt`**: Square root with SIMD optimization
//! - **`pow`**: Power function with special case optimization
//!
//! ## Matrix Operations
//! - **`matmul`**: Matrix multiplication with blocked algorithms and kernel selection
//!
//! # Design Principles
//!
//! - **Performance First**: Every operation optimized for maximum speed
//! - **Numerical Stability**: Careful implementation of mathematical functions
//! - **SIMD Optimization**: Vectorized operations where beneficial
//! - **Memory Efficiency**: Minimal allocations and cache-friendly access
//! - **Type Safety**: Strong typing and bounds checking
//! - **Modular Organization**: One operation per file for maintainability
//! - **Comprehensive Testing**: FFI validation against LibTorch reference
//! - **Zero-Cost Abstractions**: Minimal overhead for operation dispatch
//! - **Broadcasting Support**: Automatic shape compatibility for element-wise operations
//! - **Gradient Preservation**: Full gradtrack support for all operations

pub mod add;
pub mod broadcasting;
pub mod div;
pub mod exp;
pub mod leaky_relu;
pub mod log;
pub mod matmul;
pub mod mul;
pub mod pow;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod sqrt;
pub mod sub;
pub mod tanh;
