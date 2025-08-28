use super::libtorch_tensor::LibTorchTensor;
use super::{get_last_error, LibtorchAdam};

/// Safe Rust wrapper around LibTorch Adam optimizer for validation purposes.
///
/// This struct provides a safe interface to PyTorch's Adam optimizer, enabling
/// rigorous validation of optimization algorithms. It handles memory management,
/// error propagation, and provides identical behavior to PyTorch's Adam.
#[derive(Debug)]
pub struct LibTorchAdam {
    inner: *mut LibtorchAdam,
}

impl LibTorchAdam {
    /// Create a new Adam optimizer with the given parameters
    pub fn new(
        parameters: &[&LibTorchTensor],
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> Result<Self, String> {
        // Create array of parameter pointers
        let mut param_ptrs: Vec<*mut super::LibtorchTensor> =
            parameters.iter().map(|tensor| tensor.inner).collect();

        let optimizer_ptr = unsafe {
            super::libtorch_adam_new(
                param_ptrs.as_mut_ptr(),
                param_ptrs.len(),
                learning_rate,
                beta1,
                beta2,
                eps,
                weight_decay,
                amsgrad,
            )
        };

        if optimizer_ptr.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchAdam {
            inner: optimizer_ptr,
        })
    }

    /// Perform a single optimization step
    pub fn step(&self) -> Result<(), String> {
        unsafe {
            super::libtorch_adam_step(self.inner);
        }

        // Check for errors
        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Zero out all parameter gradients
    pub fn zero_grad(&self) -> Result<(), String> {
        unsafe {
            super::libtorch_adam_zero_grad(self.inner);
        }

        // Check for errors
        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Set the learning rate for all parameter groups
    #[allow(unused)]
    pub fn set_learning_rate(&self, learning_rate: f64) -> Result<(), String> {
        unsafe {
            super::libtorch_adam_set_learning_rate(self.inner, learning_rate);
        }

        // Check for errors
        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Get the current learning rate
    #[allow(unused)]
    pub fn learning_rate(&self) -> f64 {
        unsafe { super::libtorch_adam_get_learning_rate(self.inner) }
    }

    /// Get a parameter by index after optimization
    #[allow(unused)]
    pub fn get_parameter(&self, index: usize) -> Result<LibTorchTensor, String> {
        let param_ptr = unsafe { super::libtorch_adam_get_parameter(self.inner, index) };

        if param_ptr.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: param_ptr })
    }

    /// Get the number of parameters
    #[allow(unused)]
    pub fn num_parameters(&self) -> usize {
        unsafe { super::libtorch_adam_get_num_parameters(self.inner) }
    }
}

impl Drop for LibTorchAdam {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                super::libtorch_adam_free(self.inner);
            }
        }
    }
}
