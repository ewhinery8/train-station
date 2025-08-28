//! Internal FFI bindings for LibTorch integration
//!
//! This module contains the raw FFI declarations and safe wrappers for LibTorch.
//! It is not exposed in the public API and is only used internally by the
//! validation and performance modules.

use std::ffi::CStr;

// Manual FFI bindings - avoiding bindgen dependency
#[repr(C)]
pub struct LibtorchTensor {
    _private: [u8; 0],
}

#[repr(C)]
pub struct LibtorchAdam {
    _private: [u8; 0],
}

extern "C" {
    // Tensor creation and management
    fn libtorch_tensor_new(shape: *const i64, ndim: usize) -> *mut LibtorchTensor;
    fn libtorch_tensor_zeros(shape: *const i64, ndim: usize) -> *mut LibtorchTensor;
    fn libtorch_tensor_ones(shape: *const i64, ndim: usize) -> *mut LibtorchTensor;
    fn libtorch_tensor_from_data(
        data: *const f32,
        shape: *const i64,
        ndim: usize,
    ) -> *mut LibtorchTensor;

    // Tensor operations
    fn libtorch_tensor_add_scalar(
        tensor: *const LibtorchTensor,
        scalar: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_add_tensor(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_sub_scalar(
        tensor: *const LibtorchTensor,
        scalar: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_sub_tensor(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_mul_scalar(
        tensor: *const LibtorchTensor,
        scalar: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_mul_tensor(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_div_scalar(
        tensor: *const LibtorchTensor,
        scalar: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_div_tensor(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_matmul(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_exp(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_sqrt(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_log(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_pow_scalar(
        tensor: *const LibtorchTensor,
        exponent: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_pow_tensor(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_softmax(tensor: *const LibtorchTensor, dim: i64) -> *mut LibtorchTensor;
    fn libtorch_tensor_relu(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_tanh(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_sigmoid(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_leaky_relu(
        tensor: *const LibtorchTensor,
        negative_slope: f32,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_sum(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_mean(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_std(tensor: *const LibtorchTensor, unbiased: bool) -> *mut LibtorchTensor;
    fn libtorch_tensor_norm(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_var(tensor: *const LibtorchTensor, unbiased: bool) -> *mut LibtorchTensor;
    fn libtorch_tensor_sum_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_mean_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_std_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
        unbiased: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_norm_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_var_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
        unbiased: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_min(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_min_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_max(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_max_dims(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndims: usize,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_argmin(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_argmin_dim(
        tensor: *const LibtorchTensor,
        dim: i64,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_argmax(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_argmax_dim(
        tensor: *const LibtorchTensor,
        dim: i64,
        keepdim: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_cat(
        tensors: *const *const LibtorchTensor,
        count: usize,
        dim: i64,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_stack(
        tensors: *const *const LibtorchTensor,
        count: usize,
        dim: i64,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_view(
        tensor: *const LibtorchTensor,
        shape: *const i64,
        ndim: usize,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_permute(
        tensor: *const LibtorchTensor,
        dims: *const i64,
        ndim: usize,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_index_select(
        tensor: *const LibtorchTensor,
        dim: i64,
        indices: *const i64,
        nindices: usize,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_gather(
        tensor: *const LibtorchTensor,
        dim: i64,
        index_data: *const i64,
        index_shape: *const i64,
        index_ndim: usize,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_select(
        tensor: *const LibtorchTensor,
        dim: i64,
        index: i64,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_masked_fill(
        tensor: *const LibtorchTensor,
        mask: *const bool,
        numel: usize,
        value: f32,
    ) -> *mut LibtorchTensor;

    // Gradient operations (for autograd testing)
    fn libtorch_tensor_require_grad(
        tensor: *mut LibtorchTensor,
        requires_grad: bool,
    ) -> *mut LibtorchTensor;
    fn libtorch_tensor_backward(tensor: *mut LibtorchTensor, grad_output: *mut LibtorchTensor);
    fn libtorch_tensor_grad(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;

    // Tensor information
    fn libtorch_tensor_ndim(tensor: *const LibtorchTensor) -> usize;
    fn libtorch_tensor_shape(tensor: *const LibtorchTensor, shape: *mut i64);
    fn libtorch_tensor_numel(tensor: *const LibtorchTensor) -> usize;
    fn libtorch_tensor_data_ptr(tensor: *const LibtorchTensor) -> *const f32;

    // Tensor comparison (for testing)
    fn libtorch_tensor_allclose(
        a: *const LibtorchTensor,
        b: *const LibtorchTensor,
        rtol: f64,
        atol: f64,
    ) -> bool;

    // Memory management
    fn libtorch_tensor_free(tensor: *mut LibtorchTensor);

    // Gradient operations
    fn libtorch_tensor_set_grad(tensor: *mut LibtorchTensor, grad: *const LibtorchTensor);
    #[allow(dead_code)]
    fn libtorch_tensor_get_grad(tensor: *const LibtorchTensor) -> *mut LibtorchTensor;
    fn libtorch_tensor_zero_grad(tensor: *mut LibtorchTensor);
    fn libtorch_tensor_requires_grad(tensor: *mut LibtorchTensor, requires_grad: bool);

    // Adam optimizer functions
    fn libtorch_adam_new(
        parameters: *mut *mut LibtorchTensor,
        num_params: usize,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        amsgrad: bool,
    ) -> *mut LibtorchAdam;
    fn libtorch_adam_step(optimizer: *mut LibtorchAdam);
    fn libtorch_adam_zero_grad(optimizer: *mut LibtorchAdam);
    fn libtorch_adam_set_learning_rate(optimizer: *mut LibtorchAdam, learning_rate: f64);
    fn libtorch_adam_get_learning_rate(optimizer: *const LibtorchAdam) -> f64;
    fn libtorch_adam_get_parameter(
        optimizer: *const LibtorchAdam,
        index: usize,
    ) -> *mut LibtorchTensor;
    fn libtorch_adam_get_num_parameters(optimizer: *const LibtorchAdam) -> usize;
    fn libtorch_adam_free(optimizer: *mut LibtorchAdam);

    // Simple loss computation for validation
    fn libtorch_compute_mse_loss(
        prediction: *const LibtorchTensor,
        target: *const LibtorchTensor,
    ) -> *mut LibtorchTensor;
    fn libtorch_backward(loss: *mut LibtorchTensor);

    // Error handling
    fn libtorch_get_last_error() -> *const std::os::raw::c_char;
    fn libtorch_clear_error();
}

// Re-export the main types for internal use
pub use libtorch_adam::LibTorchAdam;
pub use libtorch_tensor::LibTorchTensor;

// Make get_last_error available to submodules
pub(crate) fn get_last_error() -> String {
    unsafe {
        let error_ptr = libtorch_get_last_error();
        if error_ptr.is_null() {
            return String::new();
        }

        let error_cstr = CStr::from_ptr(error_ptr);
        let error_str = error_cstr.to_string_lossy().into_owned();
        libtorch_clear_error();
        error_str
    }
}

mod libtorch_adam;
mod libtorch_tensor;
