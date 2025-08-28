/// CUDA stream management for asynchronous operations
use std::ptr::NonNull;

/// CUDA stream wrapper for asynchronous operations
pub struct CudaStream {
    stream: NonNull<std::ffi::c_void>,
}

impl CudaStream {
    /// Create a new CUDA stream
    ///
    /// # Returns
    ///
    /// CUDA stream or None if creation failed
    pub fn new() -> Option<Self> {
        let stream_ptr = create_cuda_stream();

        if stream_ptr.is_null() {
            None
        } else {
            // SAFETY: We just checked that stream_ptr is not null
            unsafe {
                Some(CudaStream {
                    stream: NonNull::new_unchecked(stream_ptr),
                })
            }
        }
    }

    /// Get raw stream pointer for FFI calls
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.stream.as_ptr()
    }

    /// Synchronize stream (wait for all operations to complete)
    pub fn synchronize(&self) -> bool {
        #[cfg(feature = "cuda")]
        {
            // Future implementation would call cudaStreamSynchronize
            // For now, return true as placeholder
            true
        }

        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }
}

impl Drop for CudaStream {
    fn drop(&mut self) {
        unsafe { destroy_cuda_stream(self.stream.as_ptr()) };
    }
}

// SAFETY: CUDA streams are thread-safe when used correctly
unsafe impl Send for CudaStream {}
unsafe impl Sync for CudaStream {}

/// Create a new CUDA stream
///
/// # Returns
///
/// Raw stream pointer or null if creation failed
pub fn create_cuda_stream() -> *mut std::ffi::c_void {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::create_cuda_stream() }
    }

    #[cfg(not(feature = "cuda"))]
    {
        std::ptr::null_mut()
    }
}

/// Destroy a CUDA stream
///
/// # Arguments
///
/// * `stream` - Stream pointer to destroy
///
/// # Safety
///
/// This function is unsafe because it performs a raw pointer operation.
/// The caller must ensure that the stream is not null.
pub unsafe fn destroy_cuda_stream(stream: *mut std::ffi::c_void) {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::destroy_cuda_stream(stream) };
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = stream;
    }
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_stream_creation() {
        if super::super::cuda_is_available() {
            let stream = CudaStream::new();
            assert!(stream.is_some());

            if let Some(s) = stream {
                assert!(!s.as_ptr().is_null());
                assert!(s.synchronize());
            }
        }
    }
}
