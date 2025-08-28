/// CUDA FFI module for GPU acceleration
///
/// This module provides the foundation for CUDA support through FFI bindings.
/// It follows the same isolation principles as the LibTorch FFI - CUDA operations
/// are accessed only through validated FFI calls, never exposed in the public API.
///
/// # Design Philosophy
///
/// - CUDA support is completely optional (enabled via "cuda" feature flag)
/// - All GPU operations go through validated FFI calls
/// - Thread-safe device management through CUDA Runtime API
/// - Memory management handled through CUDA Memory API
/// - Kernel execution coordinated through CUDA streams
/// - Error handling through CUDA error codes
///
/// # Safety
///
/// CUDA operations involve:
/// - Raw device pointers and memory management
/// - Asynchronous kernel execution
/// - Multi-GPU context switching
/// - Stream synchronization
///
/// All CUDA operations are wrapped in safe FFI calls with proper error handling.

#[cfg(feature = "cuda")]
pub mod context;

#[cfg(feature = "cuda")]
pub mod memory;

#[cfg(feature = "cuda")]
pub mod kernels;

#[cfg(feature = "cuda")]
pub mod device;

#[cfg(feature = "cuda")]
pub mod stream;

// Re-export main CUDA types when feature is enabled
#[cfg(feature = "cuda")]
pub use context::CudaContext;

#[cfg(feature = "cuda")]
pub use device::{CudaDevice, CudaDeviceProperties};

#[cfg(feature = "cuda")]
pub use memory::CudaMemory;

#[cfg(feature = "cuda")]
pub use stream::CudaStream;

/// Check if CUDA is available at runtime
///
/// This function checks if CUDA is available by attempting to initialize
/// the CUDA runtime and query device count.
///
/// # Returns
///
/// `true` if CUDA is available and at least one device is found,
/// `false` otherwise or if CUDA feature is not enabled.
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        // When CUDA feature is enabled, check actual CUDA availability
        unsafe { get_cuda_device_count() > 0 }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // When CUDA feature is disabled, always return false
        false
    }
}

/// Get the number of available CUDA devices
///
/// # Returns
///
/// Number of CUDA devices available, or 0 if CUDA is not available
/// or the feature is not enabled.
pub fn cuda_device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        unsafe { get_cuda_device_count() }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Initialize CUDA system
///
/// This function initializes the CUDA runtime and sets up the default context.
/// Must be called before any CUDA operations.
///
/// # Returns
///
/// `true` if initialization successful, `false` otherwise or if CUDA feature disabled.
pub fn initialize_cuda() -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe {
            // Initialize CUDA runtime and create default context
            cuda_ffi_initialize()
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Cleanup CUDA system
///
/// This function cleans up CUDA resources and destroys contexts.
/// Should be called when CUDA is no longer needed.
pub fn cleanup_cuda() {
    #[cfg(feature = "cuda")]
    {
        unsafe {
            cuda_ffi_cleanup();
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // No-op when CUDA is disabled
    }
}

// FFI declarations for CUDA runtime functions
// These will be implemented in future CUDA C++ wrapper similar to libtorch_wrapper.cpp
#[cfg(feature = "cuda")]
extern "C" {
    /// Initialize CUDA runtime and create default context
    fn cuda_ffi_initialize() -> bool;

    /// Cleanup CUDA resources and destroy contexts  
    fn cuda_ffi_cleanup();

    /// Get the number of CUDA devices available
    fn get_cuda_device_count() -> i32;

    /// Set the current CUDA device
    fn set_cuda_device(device_id: i32) -> bool;

    /// Get current CUDA device ID
    #[allow(dead_code)]
    fn get_current_cuda_device() -> i32;

    /// Allocate memory on CUDA device
    fn cuda_malloc(size: usize) -> *mut f32;

    /// Free memory on CUDA device
    fn cuda_free(ptr: *mut f32);

    /// Copy memory between host and device
    fn cuda_memcpy(dst: *mut f32, src: *const f32, size: usize, kind: i32) -> bool;

    /// Create CUDA stream for asynchronous operations
    fn create_cuda_stream() -> *mut std::ffi::c_void;

    /// Destroy CUDA stream
    fn destroy_cuda_stream(stream: *mut std::ffi::c_void);

    /// Launch addition kernel
    fn launch_add_kernel(
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        size: usize,
        stream: *mut std::ffi::c_void,
    ) -> bool;

    /// Launch subtraction kernel
    fn launch_sub_kernel(
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        size: usize,
        stream: *mut std::ffi::c_void,
    ) -> bool;

    /// Launch multiplication kernel
    fn launch_mul_kernel(
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        size: usize,
        stream: *mut std::ffi::c_void,
    ) -> bool;

    /// Launch matrix multiplication kernel (GEMM)
    fn launch_matmul_kernel(
        a: *const f32,
        b: *const f32,
        result: *mut f32,
        m: usize,
        n: usize,
        k: usize,
        stream: *mut std::ffi::c_void,
    ) -> bool;
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability_check() {
        // This test works regardless of actual CUDA availability
        let available = cuda_is_available();
        let device_count = cuda_device_count();

        if available {
            assert!(device_count > 0, "CUDA available but no devices found");
        } else {
            assert_eq!(device_count, 0, "CUDA not available but devices reported");
        }
    }

    #[test]
    fn test_cuda_feature_flag() {
        // When CUDA feature is enabled, these functions should be callable
        let _ = cuda_is_available();
        let _ = cuda_device_count();
        let _ = initialize_cuda();
        cleanup_cuda();
    }
}

#[cfg(all(not(feature = "cuda"), test))]
mod tests_no_cuda {
    use super::*;

    #[test]
    fn test_cuda_disabled() {
        // When CUDA feature is disabled, should always return false/0
        assert!(!cuda_is_available());
        assert_eq!(cuda_device_count(), 0);
        assert!(!initialize_cuda());

        // cleanup_cuda should be a no-op
        cleanup_cuda();
    }
}
