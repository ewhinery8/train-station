use super::{get_last_error, LibtorchTensor};
use std::ptr;
use std::slice;

/// Safe Rust wrapper around LibTorch tensors for validation purposes.
///
/// This struct provides a safe interface to PyTorch's LibTorch library, enabling
/// rigorous validation of tensor operations. It handles memory management, error
/// propagation, and provides identical operations to our native Rust implementation.
///
/// ## Memory Management
///
/// `LibTorchTensor` uses RAII principles for automatic memory management:
/// - Tensors are automatically freed when dropped
/// - Thread-safe reference counting through LibTorch
/// - No manual memory management required
///
/// ## Thread Safety
///
/// All operations are thread-safe and the struct implements `Send + Sync`,
/// allowing safe usage across thread boundaries.
pub struct LibTorchTensor {
    /// Raw pointer to the C++ LibtorchTensor wrapper
    ///
    /// This pointer is managed through RAII and should never be null
    /// during the lifetime of this struct.
    pub(crate) inner: *mut LibtorchTensor,
}

impl LibTorchTensor {
    /// Creates a new uninitialized tensor with the specified shape.
    ///
    /// The tensor values are uninitialized and may contain arbitrary data.
    /// For predictable values, use [`zeros`] or [`ones`] instead.
    ///
    /// # Arguments
    ///
    /// * `shape` - Slice of dimension sizes (e.g., `&[2, 3]` for a 2x3 tensor)
    ///
    /// # Returns
    ///
    /// * `Ok(LibTorchTensor)` - Successfully created tensor
    /// * `Err(String)` - Error message if creation failed
    pub fn new(shape: &[usize]) -> Result<Self, String> {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = unsafe { super::libtorch_tensor_new(shape_i64.as_ptr(), shape_i64.len()) };

        if tensor.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: tensor })
    }

    /// Creates a new tensor filled with zeros.
    ///
    /// All elements in the tensor will be initialized to 0.0.
    ///
    /// # Arguments
    ///
    /// * `shape` - Slice of dimension sizes
    ///
    /// # Returns
    ///
    /// * `Ok(LibTorchTensor)` - Successfully created zero-filled tensor
    /// * `Err(String)` - Error message if creation failed
    pub fn zeros(shape: &[usize]) -> Result<Self, String> {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = unsafe { super::libtorch_tensor_zeros(shape_i64.as_ptr(), shape_i64.len()) };

        if tensor.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: tensor })
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: &[usize]) -> Result<Self, String> {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = unsafe { super::libtorch_tensor_ones(shape_i64.as_ptr(), shape_i64.len()) };

        if tensor.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: tensor })
    }

    /// Create a tensor from existing data
    pub fn from_data(data: &[f32], shape: &[usize]) -> Result<Self, String> {
        let shape_i64: Vec<i64> = shape.iter().map(|&x| x as i64).collect();
        let tensor = unsafe {
            super::libtorch_tensor_from_data(data.as_ptr(), shape_i64.as_ptr(), shape_i64.len())
        };

        if tensor.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: tensor })
    }

    /// Add a scalar to the tensor
    pub fn add_scalar(&self, scalar: f32) -> Result<Self, String> {
        let result = unsafe { super::libtorch_tensor_add_scalar(self.inner as *const _, scalar) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Add two tensors together
    pub fn add_tensor(&self, other: &LibTorchTensor) -> Result<Self, String> {
        let result = unsafe {
            super::libtorch_tensor_add_tensor(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Subtract a scalar from the tensor
    pub fn sub_scalar(&self, scalar: f32) -> Result<Self, String> {
        let result = unsafe { super::libtorch_tensor_sub_scalar(self.inner as *const _, scalar) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Subtract two tensors
    pub fn sub_tensor(&self, other: &LibTorchTensor) -> Result<Self, String> {
        let result = unsafe {
            super::libtorch_tensor_sub_tensor(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Multiply tensor by scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self, String> {
        let result = unsafe { super::libtorch_tensor_mul_scalar(self.inner as *const _, scalar) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Multiply two tensors
    pub fn mul_tensor(&self, other: &LibTorchTensor) -> Result<Self, String> {
        let result = unsafe {
            super::libtorch_tensor_mul_tensor(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Divide tensor by scalar
    pub fn div_scalar(&self, scalar: f32) -> Result<Self, String> {
        let result = unsafe { super::libtorch_tensor_div_scalar(self.inner as *const _, scalar) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Divide two tensors
    pub fn div_tensor(&self, other: &LibTorchTensor) -> Result<Self, String> {
        let result = unsafe {
            super::libtorch_tensor_div_tensor(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &LibTorchTensor) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_matmul(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_exp(self.inner as *const _) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_sqrt(self.inner as *const _) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_log(self.inner as *const _) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Element-wise power with scalar exponent
    pub fn pow_scalar(&self, exponent: f32) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_pow_scalar(self.inner as *const _, exponent) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Element-wise power with tensor exponent (shapes must match)
    pub fn pow_tensor(&self, other: &LibTorchTensor) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_pow_tensor(
                self.inner as *const _ as *const _,
                other.inner as *const _,
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Softmax along a dimension
    pub fn softmax(&self, dim: usize) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_softmax(self.inner as *const _, dim as i64) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// ReLU
    pub fn relu(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_relu(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Tanh
    pub fn tanh(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_tanh(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Sigmoid
    pub fn sigmoid(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_sigmoid(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// LeakyReLU
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<LibTorchTensor, String> {
        let result =
            unsafe { super::libtorch_tensor_leaky_relu(self.inner as *const _, negative_slope) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// L2 norm of all elements
    pub fn norm(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_norm(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Sum all elements into a scalar tensor
    pub fn sum(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_sum(self.inner as *const _) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Mean of all elements
    pub fn mean(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_mean(self.inner as *const _) };

        if result.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: result })
    }

    /// Standard deviation of all elements (unbiased=false for validation parity)
    pub fn std(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_std(self.inner as *const _, false) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Variance of all elements (unbiased=false)
    pub fn var(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_var(self.inner as *const _, false) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Sum over specific dimensions
    pub fn sum_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_sum_dims(self.inner as *const _, di.as_ptr(), di.len(), keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Mean over specific dimensions
    pub fn mean_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_mean_dims(self.inner as *const _, di.as_ptr(), di.len(), keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Standard deviation over specific dimensions (unbiased=false)
    pub fn std_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_std_dims(
                self.inner as *const _,
                di.as_ptr(),
                di.len(),
                keepdim,
                false,
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// L2 norm over specific dimensions
    pub fn norm_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_norm_dims(self.inner as *const _, di.as_ptr(), di.len(), keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Variance over specific dimensions (unbiased=false)
    pub fn var_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_var_dims(
                self.inner as *const _,
                di.as_ptr(),
                di.len(),
                keepdim,
                false,
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Min over all elements
    pub fn min(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_min(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Min over specific dimensions
    pub fn min_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_min_dims(self.inner as *const _, di.as_ptr(), di.len(), keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Max over all elements
    pub fn max(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_max(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Max over specific dimensions
    pub fn max_dims(&self, dims: &[usize], keepdim: bool) -> Result<LibTorchTensor, String> {
        let di: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_max_dims(self.inner as *const _, di.as_ptr(), di.len(), keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Argmin over all elements (returns float indices shaped [1])
    pub fn argmin(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_argmin(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Argmin over a specific dimension
    pub fn argmin_dim(&self, dim: usize, keepdim: bool) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_argmin_dim(self.inner as *const _, dim as i64, keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Argmax over all elements (returns float indices shaped [1])
    pub fn argmax(&self) -> Result<LibTorchTensor, String> {
        let result = unsafe { super::libtorch_tensor_argmax(self.inner as *const _) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Argmax over a specific dimension
    pub fn argmax_dim(&self, dim: usize, keepdim: bool) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_argmax_dim(self.inner as *const _, dim as i64, keepdim)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Concatenate tensors along a given dimension
    pub fn cat(tensors: &[LibTorchTensor], dim: usize) -> Result<LibTorchTensor, String> {
        let ptrs: Vec<*const LibtorchTensor> =
            tensors.iter().map(|t| t.inner as *const _).collect();
        let result = unsafe { super::libtorch_tensor_cat(ptrs.as_ptr(), ptrs.len(), dim as i64) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Stack tensors along a new dimension
    pub fn stack(tensors: &[LibTorchTensor], dim: usize) -> Result<LibTorchTensor, String> {
        let ptrs: Vec<*const LibtorchTensor> =
            tensors.iter().map(|t| t.inner as *const _).collect();
        let result = unsafe { super::libtorch_tensor_stack(ptrs.as_ptr(), ptrs.len(), dim as i64) };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// View (reshape) a contiguous tensor to a new shape
    pub fn view(&self, new_shape: &[usize]) -> Result<LibTorchTensor, String> {
        let shape_i64: Vec<i64> = new_shape.iter().map(|&x| x as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_view(self.inner as *const _, shape_i64.as_ptr(), shape_i64.len())
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Permute (reorder) dimensions of a tensor
    pub fn permute(&self, dims: &[usize]) -> Result<LibTorchTensor, String> {
        let dims_i64: Vec<i64> = dims.iter().map(|&x| x as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_permute(
                self.inner as *const _,
                dims_i64.as_ptr(),
                dims_i64.len(),
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Index select along a dimension
    pub fn index_select(&self, dim: usize, indices: &[usize]) -> Result<LibTorchTensor, String> {
        let idx_i64: Vec<i64> = indices.iter().map(|&x| x as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_index_select(
                self.inner as *const _,
                dim as i64,
                idx_i64.as_ptr(),
                idx_i64.len(),
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Gather values along a dimension using an index tensor (flattened data + shape)
    pub fn gather(
        &self,
        dim: usize,
        index_shape: &[usize],
        indices: &[usize],
    ) -> Result<LibTorchTensor, String> {
        let ishape_i64: Vec<i64> = index_shape.iter().map(|&x| x as i64).collect();
        let idata_i64: Vec<i64> = indices.iter().map(|&x| x as i64).collect();
        let result = unsafe {
            super::libtorch_tensor_gather(
                self.inner,
                dim as i64,
                idata_i64.as_ptr(),
                ishape_i64.as_ptr(),
                ishape_i64.len(),
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Masked fill: replace values where mask is true with `value`
    pub fn masked_fill(&self, mask: &[bool], value: f32) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_masked_fill(
                self.inner as *const _,
                mask.as_ptr(),
                mask.len(),
                value,
            )
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Select along a dimension
    pub fn select(&self, dim: usize, index: usize) -> Result<LibTorchTensor, String> {
        let result = unsafe {
            super::libtorch_tensor_select(self.inner as *const _, dim as i64, index as i64)
        };
        if result.is_null() {
            return Err(get_last_error());
        }
        Ok(LibTorchTensor { inner: result })
    }

    /// Set requires_grad for autograd
    pub fn requires_grad_(self, requires_grad: bool) -> Result<Self, String> {
        let result =
            unsafe { super::libtorch_tensor_require_grad(self.inner as *mut _, requires_grad) };

        if result.is_null() {
            return Err(get_last_error());
        }

        // The C++ function returns the same tensor, so we don't change self.inner
        Ok(self)
    }

    /// Perform backward pass
    pub fn backward(&self, grad_output: Option<&LibTorchTensor>) -> Result<(), String> {
        unsafe {
            let grad_ptr = grad_output.map(|g| g.inner).unwrap_or(ptr::null_mut());
            super::libtorch_tensor_backward(self.inner as *mut _, grad_ptr);
        }

        // Check for errors
        let error = get_last_error();
        if !error.is_empty() {
            return Err(error);
        }

        Ok(())
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        unsafe { super::libtorch_tensor_ndim(self.inner as *const _) }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> Vec<usize> {
        let ndim = self.ndim();
        let mut shape = vec![0i64; ndim];
        unsafe {
            super::libtorch_tensor_shape(self.inner as *const _, shape.as_mut_ptr());
        }
        shape.into_iter().map(|x| x as usize).collect()
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        unsafe { super::libtorch_tensor_numel(self.inner as *const _) }
    }

    /// Get a slice of the tensor data
    pub fn data(&self) -> &[f32] {
        let ptr = unsafe { super::libtorch_tensor_data_ptr(self.inner as *const _) };
        if ptr.is_null() {
            return &[];
        }

        let numel = self.numel();
        unsafe { slice::from_raw_parts(ptr, numel) }
    }

    /// Check if tensors are approximately equal
    pub fn allclose(&self, other: &LibTorchTensor, rtol: f64, atol: f64) -> bool {
        unsafe {
            super::libtorch_tensor_allclose(
                self.inner as *const _ as *const _,
                other.inner as *const _,
                rtol,
                atol,
            )
        }
    }

    /// Set the gradient for this tensor
    pub fn set_grad(&self, grad: &LibTorchTensor) -> Result<(), String> {
        unsafe {
            super::libtorch_tensor_set_grad(self.inner as *mut _, grad.inner);
        }

        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Get the gradient for this tensor
    pub fn grad(&self) -> Option<LibTorchTensor> {
        let grad_ptr = unsafe { super::libtorch_tensor_grad(self.inner as *const _) };

        if grad_ptr.is_null() {
            None
        } else {
            Some(LibTorchTensor { inner: grad_ptr })
        }
    }

    /// Zero out the gradient for this tensor
    pub fn zero_grad(&self) -> Result<(), String> {
        unsafe {
            super::libtorch_tensor_zero_grad(self.inner as *mut _);
        }

        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Set whether this tensor requires gradients
    pub fn requires_grad(&self, requires_grad: bool) -> Result<(), String> {
        unsafe {
            super::libtorch_tensor_requires_grad(self.inner as *mut _, requires_grad);
        }

        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }

    /// Compute MSE loss between this tensor and a target
    pub fn mse_loss(&self, target: &LibTorchTensor) -> Result<LibTorchTensor, String> {
        let loss_ptr = unsafe {
            super::libtorch_compute_mse_loss(self.inner as *const _, target.inner as *const _)
        };

        if loss_ptr.is_null() {
            return Err(get_last_error());
        }

        Ok(LibTorchTensor { inner: loss_ptr })
    }

    /// Perform backward pass to compute gradients (simple version for scalars)
    pub fn backward_scalar(&self) -> Result<(), String> {
        unsafe {
            super::libtorch_backward(self.inner as *mut _);
        }

        let error_msg = get_last_error();
        if !error_msg.is_empty() {
            return Err(error_msg);
        }

        Ok(())
    }
}

impl Drop for LibTorchTensor {
    fn drop(&mut self) {
        if !self.inner.is_null() {
            unsafe {
                super::libtorch_tensor_free(self.inner as *mut _);
            }
        }
    }
}

// Safety: LibTorchTensor can be sent between threads
unsafe impl Send for LibTorchTensor {}
unsafe impl Sync for LibTorchTensor {}
