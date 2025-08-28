/// CUDA context management
///
/// This module provides thread-safe CUDA context management similar to the
/// device context management, but specifically for CUDA devices and contexts.
use std::sync::atomic::{AtomicI32, Ordering};

/// Global current CUDA device (default: -1 = no device set)
static CURRENT_CUDA_DEVICE: AtomicI32 = AtomicI32::new(-1);

/// CUDA context for managing device state
pub struct CudaContext {
    device_id: i32,
    previous_device: i32,
}

impl CudaContext {
    /// Create a new CUDA context for the specified device
    ///
    /// # Arguments
    ///
    /// * `device_id` - CUDA device ID to switch to
    ///
    /// # Returns
    ///
    /// CUDA context that will restore previous device when dropped
    pub fn new(device_id: i32) -> Option<Self> {
        let previous_device = get_current_cuda_device();

        if set_cuda_device(device_id) {
            Some(CudaContext {
                device_id,
                previous_device,
            })
        } else {
            None
        }
    }

    /// Get the device ID for this context
    pub fn device_id(&self) -> i32 {
        self.device_id
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        // Restore previous device when context is dropped
        if self.previous_device >= 0 {
            set_cuda_device(self.previous_device);
        }
    }
}

/// Set the current CUDA device
///
/// # Arguments
///
/// * `device_id` - CUDA device ID to set as current
///
/// # Returns
///
/// `true` if device was set successfully, `false` otherwise
pub fn set_cuda_device(device_id: i32) -> bool {
    #[cfg(feature = "cuda")]
    {
        unsafe {
            if super::set_cuda_device(device_id) {
                CURRENT_CUDA_DEVICE.store(device_id, Ordering::Relaxed);
                true
            } else {
                false
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = device_id;
        false
    }
}

/// Get the current CUDA device ID
///
/// # Returns
///
/// Current CUDA device ID, or -1 if no device is set
pub fn get_current_cuda_device() -> i32 {
    #[cfg(feature = "cuda")]
    {
        CURRENT_CUDA_DEVICE.load(Ordering::Relaxed)
    }

    #[cfg(not(feature = "cuda"))]
    {
        -1
    }
}

/// Get the number of available CUDA devices
///
/// # Returns
///
/// Number of CUDA devices, or 0 if CUDA is not available
pub fn get_cuda_device_count() -> i32 {
    #[cfg(feature = "cuda")]
    {
        unsafe { super::get_cuda_device_count() }
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

/// Execute a closure with a specific CUDA device context
///
/// # Arguments
///
/// * `device_id` - CUDA device ID to use
/// * `f` - Closure to execute with the device context
///
/// # Returns
///
/// Result of the closure, or None if device context creation failed
pub fn with_cuda_device<F, R>(device_id: i32, f: F) -> Option<R>
where
    F: FnOnce() -> R,
{
    let _context = CudaContext::new(device_id)?;
    Some(f())
}

#[cfg(all(feature = "cuda", test))]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_device_context() {
        let device_count = get_cuda_device_count();

        if device_count > 0 {
            // Test device switching with context
            let initial_device = get_current_cuda_device();

            {
                let _context = CudaContext::new(0);
                if _context.is_some() {
                    assert_eq!(get_current_cuda_device(), 0);
                }
            }

            // Device should be restored after context drop
            assert_eq!(get_current_cuda_device(), initial_device);
        }
    }

    #[test]
    fn test_with_cuda_device() {
        let device_count = get_cuda_device_count();

        if device_count > 0 {
            let result = with_cuda_device(0, || get_current_cuda_device());

            if let Some(device_id) = result {
                assert_eq!(device_id, 0);
            }
        }
    }
}
