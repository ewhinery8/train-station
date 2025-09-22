//! # Train Station
//!
//! PyTorch-inspired, zero-dependency Rust ML library focused on performance, research, and ergonomics.
//!
//! - **PyTorch-inspired API**: familiar semantics make switching effortless.
//! - **SIMD-aligned memory pool**: per-thread pools with 64/32/16-byte alignment.
//! - **Safe zero-copy views**: reshape/transpose/slice/as_strided with capacity checks and copy-on-write.
//! - **Iterator-first API**: idiomatic Rust iterators that preserve gradients.
//! - **Thread-safe GradTrack**: local/shared graphs with sharded storage; cross-thread capable.
//! - **Broadcasting**: NumPy-style element-wise and batched matmul broadcasting.
//!
//! Mission: enable low-level control and simple composition to build larger objects and next-gen architectures.
//!
//! ## Quick Start
//!
//! ```rust
//! use train_station::Tensor;
//!
//! // Parameters
//! let x = Tensor::ones(vec![2, 3]);
//! let w = Tensor::ones(vec![3, 2]).with_requires_grad();
//! let b = Tensor::zeros(vec![2]).with_requires_grad();
//!
//! // Forward and backward
//! let y = x.matmul(&w).add_tensor(&b).relu();
//! let mut loss = y.sum();
//! loss.backward(None);
//!
//! // Access gradients
//! let gw = w.grad_owned().unwrap();
//! assert_eq!(gw.shape().dims(), vec![3, 2]);
//! ```
//!
//! ## Views (zero-copy) with gradients
//!
//! ```rust
//! use train_station::Tensor;
//!
//! let x = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![4]).unwrap().with_requires_grad();
//! let v = x.slice_view(1, 1, 2); // [2.0, 3.0]
//! let mut s = v.sum();
//! s.backward(None);
//! let gx = x.grad_owned().unwrap();
//! assert_eq!(gx.data(), &[0.0, 1.0, 1.0, 0.0]);
//! ```
//!
//! ## Broadcasting
//!
//! ```rust
//! use train_station::Tensor;
//!
//! let a = Tensor::ones(vec![2, 3]);
//! let b = Tensor::ones(vec![1, 3]);
//! let c = a.add_tensor(&b); // [2,3] + [1,3] -> [2,3]
//! assert_eq!(c.shape().dims(), vec![2, 3]);
//! ```
//!
//! ## Iterators
//!
//! ```rust
//! use train_station::Tensor;
//!
//! let t = Tensor::from_slice(&(0..12).map(|x| x as f32).collect::<Vec<_>>(), vec![12]).unwrap();
//! let chunks: Vec<Tensor> = t.iter_chunks(4).collect();
//! assert_eq!(chunks.len(), 3);
//! ```
//!
//! ## Memory pool controls
//!
//! ```rust
//! use train_station::Tensor;
//! use train_station::tensor::with_no_mem_pool;
//!
//! // Default: pooled allocation
//! let a = Tensor::new(vec![64]);
//!
//! // Force system allocator within a scope (useful for diagnostics or memory studies)
//! let b = with_no_mem_pool(|| Tensor::new(vec![64]));
//! assert_eq!(a.size(), b.size());
//! ```
//!
//! Threading note: the memory pool is thread-local. If you create tensors in a worker
//! thread and return them to another thread (e.g., main), prefer wrapping the creation in
//! `with_no_mem_pool(|| ...)` so those allocations use the system allocator instead of a
//! thread-local pool.
//!
//! ## Modules
//!
//! - `tensor`: core tensor, ops (SIMD), broadcasting, views, iterators
//! - `gradtrack`: automatic differentiation (local/shared graph)
//! - `optimizers`: Adam and related utilities
//! - `serialization`: minimal JSON/binary I/O
//! - `device`: CPU today; CUDA scaffolding is feature-gated
//!
//! ## Feature Flags
//!
//! - `cuda`: compile CUDA scaffolding (experimental; CPU is the default path)
//!
//! ## Data types
//!
//! Train Station currently targets `f32` tensors. We will be expanding to additional
//! data types over time; see the Roadmap for direction.
//!
//! ### CUDA status
//!
//! The `cuda` feature is currently experimental and not ready for general use. It provides
//! scaffolding only; production use should stick to CPU. API surface may change without notice.

#[cfg(feature = "cuda")]
pub(crate) mod cuda;
pub(crate) mod device;
pub mod gradtrack;
pub mod optimizers;
pub mod serialization;
pub mod tensor;

pub use device::{
    cuda_device_count, cuda_is_available, current_device, get_default_device, set_default_device,
    with_device, Device, DeviceType,
};

pub use tensor::Tensor;
