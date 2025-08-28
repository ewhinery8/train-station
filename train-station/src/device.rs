//! Device management system for Train Station ML Library
//!
//! This module provides a unified device abstraction for CPU and CUDA operations with thread-safe
//! context management. The device system follows PyTorch's device API design while maintaining
//! zero dependencies for CPU operations and feature-gated CUDA support.
//!
//! # Design Philosophy
//!
//! The device management system is designed for:
//! - **Thread Safety**: Thread-local device contexts with automatic restoration
//! - **Zero Dependencies**: CPU operations require no external dependencies
//! - **Feature Isolation**: CUDA support is completely optional and feature-gated
//! - **PyTorch Compatibility**: Familiar API design for users coming from PyTorch
//! - **Performance**: Minimal overhead for device switching and context management
//!
//! # Organization
//!
//! The device module is organized into several key components:
//! - **Device Types**: `DeviceType` enum for CPU and CUDA device types
//! - **Device Representation**: `Device` struct with type and index information
//! - **Context Management**: Thread-local device stack with automatic restoration
//! - **Global Default**: Atomic global default device for new tensor creation
//! - **CUDA Integration**: Feature-gated CUDA availability and device count functions
//!
//! # Key Features
//!
//! - **Thread-Local Contexts**: Each thread maintains its own device context stack
//! - **Automatic Restoration**: Device contexts are automatically restored when dropped
//! - **Global Default Device**: Configurable default device for new tensor creation
//! - **CUDA Feature Gates**: All CUDA functionality is feature-gated and optional
//! - **Runtime Validation**: CUDA device indices are validated at runtime
//! - **Zero-Cost CPU Operations**: CPU device operations have no runtime overhead
//!
//! # Examples
//!
//! ## Basic Device Usage
//!
//! ```rust
//! use train_station::{Device, DeviceType};
//!
//! // Create CPU device
//! let cpu_device = Device::cpu();
//! assert!(cpu_device.is_cpu());
//! assert_eq!(cpu_device.index(), 0);
//! assert_eq!(cpu_device.to_string(), "cpu");
//!
//! // Create CUDA device (when feature enabled)
//! #[cfg(feature = "cuda")]
//! {
//!     if train_station::cuda_is_available() {
//!         let cuda_device = Device::cuda(0);
//!         assert!(cuda_device.is_cuda());
//!         assert_eq!(cuda_device.index(), 0);
//!         assert_eq!(cuda_device.to_string(), "cuda:0");
//!     }
//! }
//! ```
//!
//! ## Device Context Management
//!
//! ```rust
//! use train_station::{Device, with_device, current_device, set_default_device};
//!
//! // Get current device context
//! let initial_device = current_device();
//! assert!(initial_device.is_cpu());
//!
//! // Execute code with specific device context
//! let result = with_device(Device::cpu(), || {
//!     assert_eq!(current_device(), Device::cpu());
//!     // Device is automatically restored when closure exits
//!     42
//! });
//!
//! assert_eq!(result, 42);
//! assert_eq!(current_device(), initial_device);
//!
//! // Set global default device
//! set_default_device(Device::cpu());
//! assert_eq!(train_station::get_default_device(), Device::cpu());
//! ```
//!
//! ## CUDA Availability Checking
//!
//! ```rust
//! use train_station::{cuda_is_available, cuda_device_count, Device};
//!
//! // Check CUDA availability
//! if cuda_is_available() {
//!     let device_count = cuda_device_count();
//!     println!("CUDA available with {} devices", device_count);
//!     
//!     // Create tensors on CUDA devices
//!     for i in 0..device_count {
//!         let device = Device::cuda(i);
//!         // Use device for tensor operations
//!     }
//! } else {
//!     println!("CUDA not available, using CPU only");
//! }
//! ```
//!
//! ## Nested Device Contexts
//!
//! ```rust
//! use train_station::{Device, with_device, current_device};
//!
//! let original_device = current_device();
//!
//! // Nested device contexts are supported
//! with_device(Device::cpu(), || {
//!     assert_eq!(current_device(), Device::cpu());
//!     
//!     with_device(Device::cpu(), || {
//!         assert_eq!(current_device(), Device::cpu());
//!         // Inner context
//!     });
//!     
//!     assert_eq!(current_device(), Device::cpu());
//!     // Outer context
//! });
//!
//! // Original device is restored
//! assert_eq!(current_device(), original_device);
//! ```
//!
//! # Thread Safety
//!
//! The device management system is designed to be thread-safe:
//!
//! - **Thread-Local Contexts**: Each thread maintains its own device context stack
//! - **Atomic Global Default**: Global default device uses atomic operations for thread safety
//! - **Context Isolation**: Device contexts are isolated between threads
//! - **Automatic Cleanup**: Device contexts are automatically cleaned up when threads terminate
//! - **No Shared State**: No shared mutable state between threads for device contexts
//!
//! # Memory Safety
//!
//! The device system prioritizes memory safety:
//!
//! - **RAII Patterns**: Device contexts use RAII for automatic resource management
//! - **No Unsafe Code**: All device management code is safe Rust
//! - **Thread-Local Storage**: Uses thread-local storage for isolation
//! - **Automatic Restoration**: Device contexts are automatically restored when dropped
//! - **Feature Gates**: CUDA functionality is completely isolated when not enabled
//!
//! # Performance Characteristics
//!
//! - **Zero-Cost CPU Operations**: CPU device operations have no runtime overhead
//! - **Minimal Context Switching**: Device context switching is optimized for performance
//! - **Thread-Local Access**: Device context access is O(1) thread-local lookup
//! - **Atomic Global Default**: Global default device access uses relaxed atomic operations
//! - **Stack-Based Contexts**: Device context stack uses efficient Vec operations
//!
//! # Feature Flags
//!
//! - **`cuda`**: Enables CUDA device support and related functions
//! - **No CUDA**: When CUDA feature is disabled, all CUDA functions return safe defaults
//!
//! # Error Handling
//!
//! - **CUDA Validation**: CUDA device indices are validated at runtime
//! - **Feature Gates**: CUDA functions panic with clear messages when feature is disabled
//! - **Device Availability**: CUDA functions check device availability before use
//! - **Graceful Degradation**: System gracefully falls back to CPU when CUDA is unavailable

use std::cell::RefCell;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Device types supported by Train Station
///
/// This enum represents the different types of devices where tensor operations
/// can be performed. Currently supports CPU and CUDA GPU devices.
///
/// # Variants
///
/// * `Cpu` - CPU device for general-purpose computation
/// * `Cuda` - CUDA GPU device for accelerated computation (feature-gated)
///
/// # Examples
///
/// ```rust
/// use train_station::{DeviceType, Device};
///
/// let cpu_type = DeviceType::Cpu;
/// let cpu_device = Device::from(cpu_type);
/// assert!(cpu_device.is_cpu());
///
/// #[cfg(feature = "cuda")]
/// {
///     let cuda_type = DeviceType::Cuda;
///     let cuda_device = Device::from(cuda_type);
///     assert!(cuda_device.is_cuda());
/// }
/// ```
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared between threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// CPU device for general-purpose computation
    Cpu,
    /// CUDA GPU device for accelerated computation (feature-gated)
    Cuda,
}

impl fmt::Display for DeviceType {
    /// Format the device type as a string
    ///
    /// # Returns
    ///
    /// String representation of the device type:
    /// - `"cpu"` for CPU devices
    /// - `"cuda"` for CUDA devices
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda => write!(f, "cuda"),
        }
    }
}

/// Device representation for tensor operations
///
/// A device specifies where tensors are located and where operations should be performed.
/// Each device has a type (CPU or CUDA) and an index (0 for CPU, GPU ID for CUDA).
/// The device system provides thread-safe context management and automatic resource cleanup.
///
/// # Fields
///
/// * `device_type` - The type of device (CPU or CUDA)
/// * `index` - Device index (0 for CPU, GPU ID for CUDA)
///
/// # Examples
///
/// ```rust
/// use train_station::Device;
///
/// // Create CPU device
/// let cpu = Device::cpu();
/// assert!(cpu.is_cpu());
/// assert_eq!(cpu.index(), 0);
/// assert_eq!(cpu.to_string(), "cpu");
///
/// // Create CUDA device (when feature enabled)
/// #[cfg(feature = "cuda")]
/// {
///     if train_station::cuda_is_available() {
///         let cuda = Device::cuda(0);
///         assert!(cuda.is_cuda());
///         assert_eq!(cuda.index(), 0);
///         assert_eq!(cuda.to_string(), "cuda:0");
///     }
/// }
/// ```
///
/// # Thread Safety
///
/// This type is thread-safe and can be shared between threads. Device contexts
/// are managed per-thread using thread-local storage.
///
/// # Memory Layout
///
/// The device struct is small and efficient:
/// - Size: 16 bytes (8 bytes for enum + 8 bytes for index)
/// - Alignment: 8 bytes
/// - Copy semantics: Implements Copy for efficient passing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Device {
    device_type: DeviceType,
    index: usize,
}

impl Device {
    /// Create a CPU device
    ///
    /// CPU devices always have index 0 and are always available regardless
    /// of feature flags or system configuration.
    ///
    /// # Returns
    ///
    /// A Device representing the CPU (always index 0)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let device = Device::cpu();
    /// assert!(device.is_cpu());
    /// assert_eq!(device.index(), 0);
    /// assert_eq!(device.device_type(), train_station::DeviceType::Cpu);
    /// ```
    pub fn cpu() -> Self {
        Device {
            device_type: DeviceType::Cpu,
            index: 0,
        }
    }

    /// Create a CUDA device
    ///
    /// Creates a device representing a specific CUDA GPU. The device index
    /// must be valid for the current system configuration.
    ///
    /// # Arguments
    ///
    /// * `index` - CUDA device index (0-based)
    ///
    /// # Returns
    ///
    /// A Device representing the specified CUDA device
    ///
    /// # Panics
    ///
    /// Panics in the following cases:
    /// - CUDA feature is not enabled (`--features cuda` not specified)
    /// - CUDA is not available on the system
    /// - Device index is out of range (>= number of available devices)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// // CPU device is always available
    /// let cpu = Device::cpu();
    ///
    /// // CUDA device (when feature enabled and available)
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if train_station::cuda_is_available() {
    ///         let device_count = train_station::cuda_device_count();
    ///         if device_count > 0 {
    ///             let cuda = Device::cuda(0);
    ///             assert!(cuda.is_cuda());
    ///             assert_eq!(cuda.index(), 0);
    ///         }
    ///     }
    /// }
    /// ```
    pub fn cuda(index: usize) -> Self {
        #[cfg(feature = "cuda")]
        {
            use crate::cuda;

            // Check if CUDA is available
            if !cuda::cuda_is_available() {
                panic!("CUDA is not available on this system");
            }

            // Check if device index is valid
            let device_count = cuda::cuda_device_count();
            if index >= device_count as usize {
                panic!(
                    "CUDA device index {} out of range (0-{})",
                    index,
                    device_count - 1
                );
            }

            Device {
                device_type: DeviceType::Cuda,
                index,
            }
        }

        #[cfg(not(feature = "cuda"))]
        {
            let _ = index;
            panic!("CUDA support not enabled. Enable with --features cuda");
        }
    }

    /// Get the device type
    ///
    /// # Returns
    ///
    /// The `DeviceType` enum variant representing this device's type
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::{Device, DeviceType};
    ///
    /// let cpu = Device::cpu();
    /// assert_eq!(cpu.device_type(), DeviceType::Cpu);
    /// ```
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get the device index
    ///
    /// # Returns
    ///
    /// The device index (0 for CPU, GPU ID for CUDA)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let cpu = Device::cpu();
    /// assert_eq!(cpu.index(), 0);
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if train_station::cuda_is_available() {
    ///         let cuda = Device::cuda(0);
    ///         assert_eq!(cuda.index(), 0);
    ///     }
    /// }
    /// ```
    pub fn index(&self) -> usize {
        self.index
    }

    /// Check if this is a CPU device
    ///
    /// # Returns
    ///
    /// `true` if this device represents a CPU, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let cpu = Device::cpu();
    /// assert!(cpu.is_cpu());
    /// assert!(!cpu.is_cuda());
    /// ```
    pub fn is_cpu(&self) -> bool {
        self.device_type == DeviceType::Cpu
    }

    /// Check if this is a CUDA device
    ///
    /// # Returns
    ///
    /// `true` if this device represents a CUDA GPU, `false` otherwise
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let cpu = Device::cpu();
    /// assert!(!cpu.is_cuda());
    /// assert!(cpu.is_cpu());
    /// ```
    pub fn is_cuda(&self) -> bool {
        self.device_type == DeviceType::Cuda
    }
}

impl Default for Device {
    /// Create the default device (CPU)
    ///
    /// # Returns
    ///
    /// A CPU device (same as `Device::cpu()`)
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let device = Device::default();
    /// assert!(device.is_cpu());
    /// assert_eq!(device, Device::cpu());
    /// ```
    fn default() -> Self {
        Device::cpu()
    }
}

impl fmt::Display for Device {
    /// Format the device as a string
    ///
    /// # Returns
    ///
    /// String representation of the device:
    /// - `"cpu"` for CPU devices
    /// - `"cuda:{index}"` for CUDA devices
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::Device;
    ///
    /// let cpu = Device::cpu();
    /// assert_eq!(cpu.to_string(), "cpu");
    ///
    /// #[cfg(feature = "cuda")]
    /// {
    ///     if train_station::cuda_is_available() {
    ///         let cuda = Device::cuda(0);
    ///         assert_eq!(cuda.to_string(), "cuda:0");
    ///     }
    /// }
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.device_type {
            DeviceType::Cpu => write!(f, "cpu"),
            DeviceType::Cuda => write!(f, "cuda:{}", self.index),
        }
    }
}

impl From<DeviceType> for Device {
    /// Convert DeviceType to Device with index 0
    ///
    /// # Arguments
    ///
    /// * `device_type` - The device type to convert
    ///
    /// # Returns
    ///
    /// A Device with the specified type and index 0
    ///
    /// # Panics
    ///
    /// Panics if `device_type` is `DeviceType::Cuda` and CUDA is not available
    /// or the feature is not enabled.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use train_station::{Device, DeviceType};
    ///
    /// let cpu_type = DeviceType::Cpu;
    /// let cpu_device = Device::from(cpu_type);
    /// assert!(cpu_device.is_cpu());
    /// assert_eq!(cpu_device.index(), 0);
    /// ```
    fn from(device_type: DeviceType) -> Self {
        match device_type {
            DeviceType::Cpu => Device::cpu(),
            DeviceType::Cuda => {
                // Call Device::cuda(0) which handles the proper feature flag checking
                Device::cuda(0)
            }
        }
    }
}

// ================================================================================================
// Device Context Management
// ================================================================================================

thread_local! {
    /// Thread-local storage for device context stack
    ///
    /// Each thread maintains its own stack of device contexts. The top of the stack
    /// represents the current device context for that thread. When a new context
    /// is pushed, it becomes the current device. When a context is popped, the
    /// previous device is restored.
    ///
    /// # Thread Safety
    ///
    /// This is thread-local storage, so each thread has its own isolated stack.
    /// No synchronization is required for access within a single thread.
    static DEVICE_STACK: RefCell<Vec<Device>> = RefCell::new(vec![Device::cpu()]);
}

/// Global default device (starts as CPU)
///
/// This atomic variable stores the global default device that is used when
/// creating new tensors without an explicit device specification. The device
/// is stored as an ID for efficient atomic operations.
///
/// # Thread Safety
///
/// Uses atomic operations for thread-safe access. Multiple threads can read
/// and write the default device concurrently without data races.
static GLOBAL_DEFAULT_DEVICE: AtomicUsize = AtomicUsize::new(0); // 0 = CPU

/// Device context guard for RAII-style device switching
///
/// This struct provides automatic restoration of the previous device context
/// when it goes out of scope, similar to PyTorch's device context manager.
/// The guard ensures that device contexts are properly cleaned up even if
/// exceptions occur.
///
/// # Thread Safety
///
/// This type is not thread-safe and should not be shared between threads.
/// Each thread should create its own device context guards.
///
/// # Examples
///
/// ```rust
/// use train_station::{Device, with_device, current_device};
///
/// let original_device = current_device();
///
/// // Use with_device instead of DeviceContext::new for public API
/// with_device(Device::cpu(), || {
///     assert_eq!(current_device(), Device::cpu());
///     // Device context is automatically restored when closure exits
/// });
///
/// assert_eq!(current_device(), original_device);
/// ```
pub struct DeviceContext {
    previous_device: Device,
}

impl DeviceContext {
    /// Create a new device context guard
    ///
    /// This function switches to the specified device and creates a guard
    /// that will automatically restore the previous device when dropped.
    ///
    /// # Arguments
    ///
    /// * `device` - The device to switch to
    ///
    /// # Returns
    ///
    /// A `DeviceContext` guard that will restore the previous device when dropped
    ///
    /// # Side Effects
    ///
    /// Changes the current thread's device context to the specified device.
    fn new(device: Device) -> Self {
        let previous_device = current_device();
        set_current_device(device);

        DeviceContext { previous_device }
    }
}

impl Drop for DeviceContext {
    /// Restore the previous device context when the guard is dropped
    ///
    /// This ensures that device contexts are properly cleaned up even if
    /// exceptions occur or the guard is dropped early.
    fn drop(&mut self) {
        set_current_device(self.previous_device);
    }
}

/// Set the global default device
///
/// This affects the default device for new tensors created without an explicit device.
/// It does not affect the current thread's device context.
///
/// # Arguments
///
/// * `device` - The device to set as the global default
///
/// # Thread Safety
///
/// This function is thread-safe and uses atomic operations to update the global default.
///
/// # Examples
///
/// ```rust
/// use train_station::{Device, set_default_device, get_default_device};
///
/// // Set global default to CPU
/// set_default_device(Device::cpu());
/// assert_eq!(get_default_device(), Device::cpu());
///
/// // The global default affects new tensor creation
/// // (tensor creation would use this default device)
/// ```
pub fn set_default_device(device: Device) {
    let device_id = device_to_id(device);
    GLOBAL_DEFAULT_DEVICE.store(device_id, Ordering::Relaxed);
}

/// Get the global default device
///
/// # Returns
///
/// The current global default device
///
/// # Thread Safety
///
/// This function is thread-safe and uses atomic operations to read the global default.
///
/// # Examples
///
/// ```rust
/// use train_station::{Device, get_default_device, set_default_device};
///
/// let initial_default = get_default_device();
/// assert!(initial_default.is_cpu());
///
/// set_default_device(Device::cpu());
/// assert_eq!(get_default_device(), Device::cpu());
/// ```
pub fn get_default_device() -> Device {
    let device_id = GLOBAL_DEFAULT_DEVICE.load(Ordering::Relaxed);
    id_to_device(device_id)
}

/// Get the current thread's device context
///
/// # Returns
///
/// The current device context for this thread
///
/// # Thread Safety
///
/// This function is thread-safe and returns the device context for the current thread only.
///
/// # Examples
///
/// ```rust
/// use train_station::{Device, current_device, with_device};
///
/// let initial_device = current_device();
/// assert!(initial_device.is_cpu());
///
/// with_device(Device::cpu(), || {
///     assert_eq!(current_device(), Device::cpu());
/// });
///
/// assert_eq!(current_device(), initial_device);
/// ```
pub fn current_device() -> Device {
    DEVICE_STACK.with(|stack| stack.borrow().last().copied().unwrap_or_else(Device::cpu))
}

/// Set the current thread's device context
///
/// This function updates the current thread's device context. It modifies the
/// top of the thread-local device stack.
///
/// # Arguments
///
/// * `device` - The device to set as the current context
///
/// # Thread Safety
///
/// This function is thread-safe and only affects the current thread's context.
///
/// # Side Effects
///
/// Changes the current thread's device context to the specified device.
fn set_current_device(device: Device) {
    DEVICE_STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if stack.is_empty() {
            stack.push(device);
        } else {
            // Replace the top of the stack
            if let Some(last) = stack.last_mut() {
                *last = device;
            }
        }
    });
}

/// Execute a closure with a specific device context
///
/// This function temporarily switches to the specified device for the duration
/// of the closure, then automatically restores the previous device. This is
/// the recommended way to execute code with a specific device context.
///
/// # Arguments
///
/// * `device` - The device to use for the closure
/// * `f` - The closure to execute
///
/// # Returns
///
/// The result of the closure
///
/// # Thread Safety
///
/// This function is thread-safe and only affects the current thread's context.
///
/// # Examples
///
/// ```rust
/// use train_station::{Device, with_device, current_device};
///
/// let original_device = current_device();
///
/// let result = with_device(Device::cpu(), || {
///     assert_eq!(current_device(), Device::cpu());
///     // Perform operations with CPU device
///     42
/// });
///
/// assert_eq!(result, 42);
/// assert_eq!(current_device(), original_device);
/// ```
pub fn with_device<F, R>(device: Device, f: F) -> R
where
    F: FnOnce() -> R,
{
    let _context = DeviceContext::new(device);
    f()
}

// Helper functions for device ID conversion
/// Convert a device to a numeric ID for storage
///
/// # Arguments
///
/// * `device` - The device to convert
///
/// # Returns
///
/// A numeric ID representing the device:
/// - 0 for CPU devices
/// - 1000 + index for CUDA devices
///
/// # Thread Safety
///
/// This function is thread-safe and has no side effects.
fn device_to_id(device: Device) -> usize {
    match device.device_type {
        DeviceType::Cpu => 0,
        DeviceType::Cuda => 1000 + device.index, // Offset CUDA devices by 1000
    }
}

/// Convert a numeric ID back to a device
///
/// # Arguments
///
/// * `id` - The numeric ID to convert
///
/// # Returns
///
/// A device representing the ID:
/// - ID 0 → CPU device
/// - ID >= 1000 → CUDA device with index (ID - 1000)
/// - Invalid IDs → CPU device (fallback)
///
/// # Thread Safety
///
/// This function is thread-safe and has no side effects.
fn id_to_device(id: usize) -> Device {
    if id == 0 {
        Device::cpu()
    } else if id >= 1000 {
        Device::cuda(id - 1000)
    } else {
        Device::cpu() // Fallback to CPU for invalid IDs
    }
}

// ================================================================================================
// CUDA Availability Functions (Direct delegation to cuda_ffi)
// ================================================================================================

/// Check if CUDA is available
///
/// This function checks if CUDA is available on the current system and
/// at least one CUDA device is found. The result depends on the CUDA
/// feature flag and system configuration.
///
/// # Returns
///
/// - `true` if CUDA feature is enabled and at least one CUDA device is available
/// - `false` if CUDA feature is disabled or no CUDA devices are found
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads.
///
/// # Examples
///
/// ```rust
/// use train_station::cuda_is_available;
///
/// if cuda_is_available() {
///     println!("CUDA is available");
///     // Create CUDA tensors and perform GPU operations
/// } else {
///     println!("CUDA is not available, using CPU only");
///     // Fall back to CPU operations
/// }
/// ```
pub fn cuda_is_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::cuda_is_available()
    }

    #[cfg(not(feature = "cuda"))]
    {
        false
    }
}

/// Get the number of CUDA devices available
///
/// This function returns the number of CUDA devices available on the system.
/// The result depends on the CUDA feature flag and system configuration.
///
/// # Returns
///
/// Number of CUDA devices available:
/// - 0 if CUDA feature is disabled
/// - 0 if CUDA is not available on the system
/// - Number of available CUDA devices if CUDA is available
///
/// # Thread Safety
///
/// This function is thread-safe and can be called from multiple threads.
///
/// # Examples
///
/// ```rust
/// use train_station::{cuda_device_count, Device};
///
/// let device_count = cuda_device_count();
/// println!("Found {} CUDA devices", device_count);
///
/// for i in 0..device_count {
///     let device = Device::cuda(i);
///     println!("CUDA device {}: {}", i, device);
/// }
/// ```
#[allow(unused)]
pub fn cuda_device_count() -> usize {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::cuda_device_count() as usize
    }

    #[cfg(not(feature = "cuda"))]
    {
        0
    }
}

// ================================================================================================
// Tests
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_device() {
        let device = Device::cpu();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert_eq!(device.index(), 0);
        assert!(device.is_cpu());
        assert!(!device.is_cuda());
        assert_eq!(device.to_string(), "cpu");
    }

    #[test]
    fn test_device_default() {
        let device = Device::default();
        assert_eq!(device.device_type(), DeviceType::Cpu);
        assert!(device.is_cpu());
    }

    #[test]
    fn test_device_type_display() {
        assert_eq!(DeviceType::Cpu.to_string(), "cpu");
        assert_eq!(DeviceType::Cuda.to_string(), "cuda");
    }

    #[test]
    fn test_device_from_device_type() {
        let device = Device::from(DeviceType::Cpu);
        assert!(device.is_cpu());
        assert_eq!(device.index(), 0);
    }

    #[test]
    #[should_panic(expected = "CUDA support not enabled. Enable with --features cuda")]
    fn test_cuda_device_panics() {
        Device::cuda(0);
    }

    #[test]
    #[should_panic(expected = "CUDA support not enabled. Enable with --features cuda")]
    fn test_device_from_cuda_type_panics() {
        let _ = Device::from(DeviceType::Cuda);
    }

    #[test]
    fn test_device_equality() {
        let cpu1 = Device::cpu();
        let cpu2 = Device::cpu();
        assert_eq!(cpu1, cpu2);
    }

    // Context management tests
    #[test]
    fn test_current_device() {
        assert_eq!(current_device(), Device::cpu());
    }

    #[test]
    fn test_default_device() {
        let initial_default = get_default_device();
        assert_eq!(initial_default, Device::cpu());

        // Should still be CPU after setting it explicitly
        set_default_device(Device::cpu());
        assert_eq!(get_default_device(), Device::cpu());
    }

    #[test]
    fn test_device_context_guard() {
        let original_device = current_device();

        {
            let _guard = DeviceContext::new(Device::cpu());
            assert_eq!(current_device(), Device::cpu());
        }

        // Device should be restored after guard is dropped
        assert_eq!(current_device(), original_device);
    }

    #[test]
    fn test_with_device() {
        let original_device = current_device();

        let result = with_device(Device::cpu(), || {
            assert_eq!(current_device(), Device::cpu());
            42
        });

        assert_eq!(result, 42);
        assert_eq!(current_device(), original_device);
    }

    #[test]
    fn test_nested_device_contexts() {
        let original = current_device();

        with_device(Device::cpu(), || {
            assert_eq!(current_device(), Device::cpu());

            with_device(Device::cpu(), || {
                assert_eq!(current_device(), Device::cpu());
            });

            assert_eq!(current_device(), Device::cpu());
        });

        assert_eq!(current_device(), original);
    }

    #[test]
    fn test_device_id_conversion() {
        assert_eq!(device_to_id(Device::cpu()), 0);
        assert_eq!(id_to_device(0), Device::cpu());

        // Test invalid ID fallback
        assert_eq!(id_to_device(999), Device::cpu());
    }

    #[test]
    fn test_cuda_availability_check() {
        // These functions should be callable regardless of CUDA availability
        let available = cuda_is_available();
        let device_count = cuda_device_count();

        if available {
            assert!(device_count > 0, "CUDA available but no devices found");
        } else {
            assert_eq!(device_count, 0, "CUDA not available but devices reported");
        }
    }
}
