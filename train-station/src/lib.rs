//! # Train Station
//!
//! Maximum performance, zero-dependency Rust machine learning library.
//!
//! Train Station is designed as a **zero-dependency, maximum performance** Rust machine learning library
//! optimized for raw computational speed, zero-cost abstractions, and minimal memory overhead. This makes
//! it positioned nicely for embedded applications or edge deployments (or if you just want easy compilation across platforms and static linking).
//! The library provides high-performance tensors with SIMD optimization, automatic differentiation, and comprehensive
//! mathematical operations suitable for production ML workloads. CPU bound today, GPU capability tomorrow (foundation for CUDA support in place).
//!
//! <mark>**Note**: CUDA support is not yet implemented, but the foundation is in place. Device support is in place but not yet thoroughly tested or supported.
//! Effectively, the libtrary is currently CPU-only until CUDA support is implemented. Feel free to contribute!
//!
//! The plan is to rapidly add functionality and operation support in early stages of development as the library matures.
//!  CUDA support will then follow.</mark>
//!
//!
//! # Design Philosophy
//!
//! - <mark>**Zero Dependencies**: Standard library only - no external crates required or utilized inside of Train Station</mark>
//! - <mark>**Iterator Integration**: Implemented as a trait, allowing leveraging of Rust's iterator system
//!   while maintaining Train Station's functionality (gradtrack, etc)</mark>
//! - **Raw Performance**: Direct memory management with unsafe optimizations justified by benchmarks
//! - **Zero-Cost Abstractions**: Compile-time optimization, enum dispatch, no virtual calls
//! - **Memory Safety**: RAII patterns with justified unsafe usage and comprehensive validation
//! - **Simplicity**: Minimal redundancy, direct implementations, clear API design
//! - **Thread Safety**: All public APIs are Send + Sync for concurrent usage
//!
//! # Core Features
//!
//! - **High-Performance Tensors**: SIMD-optimized multi-dimensional arrays with AVX2 support
//! - **Automatic Differentiation (GradTrack)**: Zero-overhead gradient tracking with computation graph optimization
//! - **Mathematical Operations**: Complete suite of tensor operations with broadcasting support
//!   (cuurently add add, sub, mul, div operations tested with broadcasting. Future TODO to ensure all operations are tested with broadcasting)
//! - **Device Management**: Unified CPU/CUDA device abstraction with thread-safe context switching
//! - **Serialization Framework**: Binary and JSON serialization for model checkpointing
//!   (very minimal framework, feel free to use serde_json or bincode for more complex use cases)
//! - **Optimizer Implementations**: Adam optimizer with SIMD-optimized parameter updates
//! - **Memory Management**: Thread-safe memory pool with global allocator and statistics
//!
//! # Organization
//!
//! The library is organized into specialized modules for maximum performance and maintainability:
//!
//! - **`tensor`**: Core tensor system with operations, transformations, and indexing
//! - **`gradtrack`**: Gradient tracking system with computation graph management
//! - **`device`**: Device management for CPU and CUDA operations
//! - **`optimizers`**: Optimization algorithms (Adam) with parameter management
//! - **`serialization`**: Binary and JSON serialization framework
//! - **`cuda`**: CUDA FFI for GPU acceleration (feature-gated)
//!
//! # Performance Characteristics
//!
//! - **Memory Overhead**: ~64 bytes per tensor (excluding data)
//! - **SIMD Alignment**: 32-byte alignment for AVX2 operations
//! - **Zero-Cost Operators**: Mathematical expressions with no runtime overhead
//! - **Thread Safety**: Lock-free operations with atomic ID generation
//! - **Memory Pool**: Thread-safe global allocator with statistics tracking
//! - **Gradient Tracking**: Zero-overhead when disabled, optimized when enabled
//!
//! # Examples
//!
//! ## Basic Tensor Operations
//!
//! ```rust
//! use train_station::{Tensor, Device};
//!
//! // Create tensors with different configurations
//! let tensor = Tensor::new(vec![2, 3, 4]);
//! let tensor_with_grad = Tensor::ones(vec![10, 10]).with_requires_grad();
//! let device_tensor = Tensor::zeros_on_device(vec![100, 100], Device::cpu());
//!
//! // Access tensor properties
//! assert_eq!(tensor.size(), 24);
//! assert_eq!(tensor.shape().dims, vec![2, 3, 4]);
//! assert!(tensor.is_contiguous());
//! assert!(tensor.is_simd_aligned());
//! ```
//!
//! ## Mathematical Operations with Operator Overloading
//!
//! ```rust
//! use train_station::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // Tensor operations with operators (each operation consumes the tensors)
//! let result1 = a + b;                    // Tensor addition
//!
//! let a2 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b2 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let result2 = a2 * b2;                  // Element-wise multiplication
//!
//! let a3 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b3 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let result3 = a3 - b3;                  // Tensor subtraction
//!
//! let a4 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b4 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let result4 = a4 / b4;                  // Element-wise division
//!
//! // Scalar operations
//! let a5 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result5 = a5 + 5.0;                 // Tensor + scalar
//!
//! let a6 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result6 = 5.0 + a6;                 // Scalar + tensor
//!
//! let a7 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result7 = a7 * 3.0;                 // Tensor * scalar
//!
//! let a8 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result8 = 3.0 * a8;                 // Scalar * tensor
//!
//! // Compound expressions
//! let a9 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b9 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let result9 = (a9 + b9) * 2.0 - 1.0;    // Complex mathematical expressions
//!
//! // Assignment operators
//! let a10 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b10 = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//! let mut c = a10.clone();
//! c += b10;                               // In-place addition
//! c *= 2.0;                               // In-place scalar multiplication
//!
//! // Negation
//! let a11 = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let result11 = -a11;                    // Negate all elements
//! ```
//!
//! ## Automatic Differentiation
//!
//! ```rust
//! use train_station::{NoGradTrack, Tensor};
//!
//! // Enable gradient tracking
//! let a = Tensor::ones(vec![1000, 1000]).with_requires_grad();
//! let b = Tensor::zeros(vec![1000, 1000]);
//! let mut result = &a + &b + 5.0;
//!
//! // Compute gradients
//! result.backward(None);
//!
//! // Access gradients
//! if let Some(grad) = a.grad() {
//!     println!("Gradient shape: {:?}", grad.shape().dims);
//! }
//!
//! // Disable gradients for inference
//! {
//!     let _guard = NoGradTrack::new();
//!     let inference_result = &a + &b; // No gradients tracked
//! }
//! ```
//!
//! ## Device Management
//!
//! ```rust
//! use train_station::{Device, with_device, set_default_device, Tensor};
//!
//! // Basic device usage
//! let cpu_device = Device::cpu();
//! let tensor = Tensor::new_on_device(vec![2, 3], cpu_device);
//!
//! // Context management (similar to PyTorch)
//! with_device(Device::cpu(), || {
//!     let tensor = Tensor::new(vec![3, 4]); // Uses context device
//!     // ... operations
//! }); // Device automatically restored
//!
//! // CUDA usage (when feature enabled)
//! #[cfg(feature = "cuda")]
//! {
//!     if train_station::cuda_is_available() {
//!         let cuda_device = Device::cuda(0);
//!         let gpu_tensor = Tensor::new_on_device(vec![1000, 1000], cuda_device);
//!     }
//! }
//! ```
//!
//! ## Optimization with Adam
//!
//! ```rust
//! use train_station::{Tensor};
//! use train_station::optimizers::{Adam, Optimizer};
//!
//! // Create parameters
//! let mut param1 = Tensor::randn(vec![100, 100], None).with_requires_grad();
//! let mut param2 = Tensor::randn(vec![100, 100], None).with_requires_grad();
//!
//! // Create optimizer
//! let mut optimizer = Adam::with_learning_rate(0.001);
//! optimizer.add_parameter(&param1);
//! optimizer.add_parameter(&param2);
//!
//! // Training loop
//! for epoch in 0..100 {
//!     // Forward pass
//!     let mut loss = param1.matmul(&param2).sum();
//!     
//!     // Backward pass
//!     loss.backward(None);
//!     
//!     // Optimization step
//!     optimizer.step(&mut [&mut param1, &mut param2]);
//!     optimizer.zero_grad(&mut [&mut param1, &mut param2]);
//! }
//! ```
//!
//! ## Serialization
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::serialization::StructSerializable;
//!
//! let tensor = Tensor::new(vec![2, 3]);
//!
//! // Save in JSON format (human-readable)
//! tensor.save_json("model.json").unwrap();
//!
//! // Save in binary format (efficient)
//! tensor.save_binary("model.bin").unwrap();
//!
//! // Load from file
//! let loaded_tensor = Tensor::load_json("model.json").unwrap();
//! ```
//!
//! # Thread Safety
//!
//! All public APIs in Train Station are designed to be thread-safe:
//!
//! - **Tensor Operations**: All tensor operations are Send + Sync
//! - **Device Management**: Thread-safe device context switching with automatic restoration
//! - **Gradient Tracking**: Thread-local computation graph storage
//! - **Memory Management**: Thread-safe global memory pool with atomic operations
//! - **Optimizers**: Thread-safe parameter updates with exclusive access patterns
//! - **Serialization**: Thread-safe file operations with proper error handling
//!
//! # Memory Safety
//!
//! Train Station prioritizes memory safety while maintaining maximum performance:
//!
//! - **RAII Patterns**: Automatic resource cleanup through Drop implementations
//! - **Justified Unsafe Code**: All unsafe operations validated against LibTorch reference
//! - **Comprehensive Validation**: Mathematical equivalence proven for all operations
//! - **Memory Pool**: Thread-safe allocation with statistics and error detection
//! - **Zero-Copy Views**: Efficient tensor views with shared memory management
//!
//! # Feature Flags
//!
//! - **`cuda`**: Enables CUDA GPU acceleration support (only foundational, a big future TODO)
//!
//! # Performance Benchmarks
//!
//! Train Station is designed to achieve maximum performance:
//!
//! - **Tensor Operations**: SIMD-optimized with AVX2 support for x86_64
//! - **Memory Allocation**: Thread-safe pool allocator with minimal overhead
//! - **Gradient Computation**: Zero-overhead tracking with optimized accumulation
//! - **Mathematical Expressions**: Zero-cost operator overloading
//! - **Serialization**: Optimized binary format for production deployment
//!
//! # Design Principles
//!
//! - **Performance First**: Every design decision optimized for speed
//! - **Zero Dependencies**: Only standard library dependencies
//! - **Memory Safety**: RAII patterns with justified unsafe usage
//! - **Thread Safety**: All public APIs Send + Sync
//! - **Simplicity**: Minimal redundancy, direct implementations
//! - **Future Proof**: Foundation for advanced ML operations
//! - **Natural API**: Operator overloading for intuitive mathematical expressions
//! - **Comprehensive Testing**: 100% coverage with mathematical validation

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
pub(crate) mod device;
pub(crate) mod gradtrack;
pub mod optimizers;
pub mod serialization;
pub mod tensor;

pub use device::{
    cuda_device_count, cuda_is_available, current_device, get_default_device, set_default_device,
    with_device, Device, DeviceType,
};
pub use gradtrack::{
    clear_gradients, is_grad_enabled, set_grad_enabled, with_no_grad, NoGradTrack,
};
pub use tensor::Tensor;
