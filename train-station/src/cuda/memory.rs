/// CUDA memory management
///
/// This module provides safe wrappers around CUDA memory allocation and management.
use std::ptr::NonNull;

/// CUDA memory allocation wrapper
pub struct CudaMemory {
    ptr: NonNull<f32>,
    size: usize,
}

impl CudaMemory {
    /// Allocate memory on CUDA device
    ///
    /// # Arguments
    ///
    /// * `size` - Number of f32 elements to allocate
    ///
    /// # Returns
    ///
    /// CUDA memory allocation or None if allocation failed
    pub fn new(size: usize) -> Option<Self> {
        let ptr = cuda_malloc(size);

        if ptr.is_null() {
            None
        } else {
            // SAFETY: We just checked that ptr is not null
            unsafe {
                Some(CudaMemory {
                    ptr: NonNull::new_unchecked(ptr),
                    size,
                })
            }
        }
    }

    /// Get raw device pointer
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// Get mutable raw device pointer
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    /// Get size in number of elements
    pub fn size(&self) -> usize {
        self.size
    }

    /// Copy data from host to device
    ///
    /// # Arguments
    ///
    /// * `host_data` - Host data to copy
    ///
    /// # Returns
    ///
    /// `true` if copy was successful, `false` otherwise
    pub fn copy_from_host(&mut self, host_data: &[f32]) -> bool {
        assert!(
            host_data.len() <= self.size,
            "Host data exceeds allocated size"
        );

        unsafe {
            cuda_memcpy(
                self.as_mut_ptr(),
                host_data.as_ptr(),
                host_data.len(),
                0, // cudaMemcpyHostToDevice
            )
        }
    }

    /// Copy data from device to host
    ///
    /// # Arguments
    ///
    /// * `host_data` - Host buffer to copy to
    ///
    /// # Returns
    ///
    /// `true` if copy was successful, `false` otherwise
    pub fn copy_to_host(&self, host_data: &mut [f32]) -> bool {
        assert!(host_data.len() <= self.size, "Host buffer too small");

        unsafe {
            cuda_memcpy(
                host_data.as_mut_ptr(),
                self.as_ptr(),
                host_data.len(),
                1, // cudaMemcpyDeviceToHost
            )
        }
    }
}

impl Drop for CudaMemory {
    fn drop(&mut self) {
        unsafe { cuda_free(self.ptr.as_ptr()) };
    }
}

// SAFETY: CudaMemory can be safely transferred between threads
// The underlying CUDA memory is managed by CUDA runtime
unsafe impl Send for CudaMemory {}
unsafe impl Sync for CudaMemory {}

/// Allocate memory on CUDA device
///
/// # Arguments
///
/// * `size` - Number of f32 elements to allocate
///
/// # Returns
///
/// Raw device pointer or null if allocation failed
pub fn cuda_malloc(size: usize) -> *mut f32 {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::cuda_malloc(size) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = size;
        std::ptr::null_mut()
    }
}

/// Free memory on CUDA device
///
/// # Arguments
///
/// * `ptr` - Device pointer to free
///
/// # Safety
///
/// This function is unsafe because it performs a raw pointer operation.
/// The caller must ensure that the ptr is not null.
pub unsafe fn cuda_free(ptr: *mut f32) {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::cuda_free(ptr) };
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = ptr;
    }
}

/// Copy memory between host and device
///
/// # Arguments
///
/// * `dst` - Destination pointer
/// * `src` - Source pointer  
/// * `size` - Number of elements to copy
/// * `kind` - Copy direction (0=HostToDevice, 1=DeviceToHost, 2=DeviceToDevice)
///
/// # Returns
///
/// `true` if copy was successful, `false` otherwise
///
/// # Safety
///
/// This function is unsafe because it performs a raw pointer operation.
/// The caller must ensure that the pointers are valid and that the size is correct.
/// The caller must also ensure that the kind is valid.
/// The caller must also ensure that the dst and src are not the same pointer.
/// The caller must also ensure that the dst and src are not null.
pub unsafe fn cuda_memcpy(dst: *mut f32, src: *const f32, size: usize, kind: i32) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::cuda_memcpy(dst, src, size, kind) }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (dst, src, size, kind);
        false
    }
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_memory_allocation() {
        let memory = CudaMemory::new(1024);

        if super::super::cuda_is_available() {
            assert!(memory.is_some());

            if let Some(mem) = memory {
                assert_eq!(mem.size(), 1024);
                assert!(!mem.as_ptr().is_null());
            }
        }
    }

    #[test]
    fn test_host_device_copy() {
        if super::super::cuda_is_available() {
            let mut memory = CudaMemory::new(10);

            if let Some(ref mut mem) = memory {
                let host_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
                assert!(mem.copy_from_host(&host_data));

                let mut result = vec![0.0; 5];
                assert!(mem.copy_to_host(&mut result));

                // Note: In actual implementation, this would verify the copy worked
                // For now, we just test that the operations don't fail
            }
        }
    }
}
