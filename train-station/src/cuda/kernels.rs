/// CUDA kernel launches for tensor operations
///
/// This module provides safe wrappers around CUDA kernel launches for
/// common tensor operations like addition, subtraction, multiplication, and matrix multiplication.
use super::stream::CudaStream;

/// Launch CUDA addition kernel
///
/// # Arguments
///
/// * `a` - First input tensor device pointer
/// * `b` - Second input tensor device pointer  
/// * `result` - Output tensor device pointer
/// * `size` - Number of elements
/// * `stream` - CUDA stream for asynchronous execution (optional)
///
/// # Safety
///
/// The caller must ensure:
/// * All pointers are valid CUDA device memory pointers
/// * `size` matches the allocated memory for all pointers
/// * No concurrent access to the same memory regions
/// * CUDA context is properly initialized when feature is enabled
///
/// # Returns
///
/// `true` if kernel launch was successful, `false` otherwise
pub unsafe fn launch_add_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_ptr());

        unsafe { super::launch_add_kernel(a, b, result, size, stream_ptr) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, result, size, stream);
        false
    }
}

/// Launch CUDA subtraction kernel
///
/// # Arguments
///
/// * `a` - First input tensor device pointer
/// * `b` - Second input tensor device pointer
/// * `result` - Output tensor device pointer
/// * `size` - Number of elements
/// * `stream` - CUDA stream for asynchronous execution (optional)
///
/// # Safety
///
/// The caller must ensure:
/// * All pointers are valid CUDA device memory pointers
/// * `size` matches the allocated memory for all pointers
/// * No concurrent access to the same memory regions
/// * CUDA context is properly initialized when feature is enabled
///
/// # Returns
///
/// `true` if kernel launch was successful, `false` otherwise  
pub unsafe fn launch_sub_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_ptr());

        unsafe { super::launch_sub_kernel(a, b, result, size, stream_ptr) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, result, size, stream);
        false
    }
}

/// Launch CUDA multiplication kernel
///
/// # Arguments
///
/// * `a` - First input tensor device pointer
/// * `b` - Second input tensor device pointer
/// * `result` - Output tensor device pointer
/// * `size` - Number of elements
/// * `stream` - CUDA stream for asynchronous execution (optional)
///
/// # Safety
///
/// The caller must ensure:
/// * All pointers are valid CUDA device memory pointers
/// * `size` matches the allocated memory for all pointers
/// * No concurrent access to the same memory regions
/// * CUDA context is properly initialized when feature is enabled
///
/// # Returns
///
/// `true` if kernel launch was successful, `false` otherwise
pub unsafe fn launch_mul_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    size: usize,
    stream: Option<&CudaStream>,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_ptr());

        unsafe { super::launch_mul_kernel(a, b, result, size, stream_ptr) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, result, size, stream);
        false
    }
}

/// Launch CUDA matrix multiplication kernel (GEMM)
///
/// # Arguments
///
/// * `a` - First matrix device pointer (M x K)
/// * `b` - Second matrix device pointer (K x N)
/// * `result` - Output matrix device pointer (M x N)
/// * `m` - Number of rows in A and result
/// * `n` - Number of columns in B and result
/// * `k` - Number of columns in A and rows in B
/// * `stream` - CUDA stream for asynchronous execution (optional)
///
/// # Safety
///
/// The caller must ensure:
/// * All pointers are valid CUDA device memory pointers
/// * Matrix dimensions match allocated memory for all pointers
/// * No concurrent access to the same memory regions
/// * CUDA context is properly initialized when feature is enabled
///
/// # Returns
///
/// `true` if kernel launch was successful, `false` otherwise
pub unsafe fn launch_matmul_kernel(
    a: *const f32,
    b: *const f32,
    result: *mut f32,
    m: usize,
    n: usize,
    k: usize,
    stream: Option<&CudaStream>,
) -> bool {
    #[cfg(feature = "cuda")]
    {
        let stream_ptr = stream.map_or(std::ptr::null_mut(), |s| s.as_ptr());

        unsafe { super::launch_matmul_kernel(a, b, result, m, n, k, stream_ptr) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (a, b, result, m, n, k, stream);
        false
    }
}

/// High-level tensor operation dispatcher
///
/// This trait provides a high-level interface for launching CUDA kernels
/// with proper error handling and stream management.
pub trait CudaKernelDispatcher {
    /// Launch element-wise addition on GPU
    fn cuda_add(&self, other: &Self, stream: Option<&CudaStream>) -> Option<Self>
    where
        Self: Sized;

    /// Launch element-wise subtraction on GPU
    fn cuda_sub(&self, other: &Self, stream: Option<&CudaStream>) -> Option<Self>
    where
        Self: Sized;

    /// Launch element-wise multiplication on GPU
    fn cuda_mul(&self, other: &Self, stream: Option<&CudaStream>) -> Option<Self>
    where
        Self: Sized;

    /// Launch matrix multiplication on GPU
    fn cuda_matmul(&self, other: &Self, stream: Option<&CudaStream>) -> Option<Self>
    where
        Self: Sized;
}

// Future implementation note:
// The CudaKernelDispatcher trait would be implemented for Tensor when CUDA support is added
// This provides a clean interface for GPU operations while keeping CUDA completely optional

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_launch_interface() {
        if super::super::cuda_is_available() {
            // Test that kernel launch functions are callable
            // In actual implementation, this would create device memory and test real kernels

            let result = unsafe {
                launch_add_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    None,
                )
            };

            // Should fail with null pointers, but function should be callable
            assert!(!result);
        }
    }

    #[test]
    fn test_all_kernel_interfaces() {
        if super::super::cuda_is_available() {
            // Test that all kernel launch functions are available
            let _ = unsafe {
                launch_add_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    None,
                )
            };
            let _ = unsafe {
                launch_sub_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    None,
                )
            };
            let _ = unsafe {
                launch_mul_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    None,
                )
            };
            let _ = unsafe {
                launch_matmul_kernel(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    0,
                    0,
                    0,
                    None,
                )
            };
        }
    }
}
