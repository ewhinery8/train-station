//! Gradient function enumeration and dispatch system for Train Station
//!
//! This module provides the core gradient function system that enables automatic differentiation
//! through zero-cost enum-based dispatch. The `GradFn` enum represents all supported tensor
//! operations and their corresponding gradient computation logic, eliminating virtual function
//! overhead while maintaining type safety and performance.
//!
//! # Architecture
//!
//! The gradient function system is organized into specialized submodules:
//! - **`ops/`** - Basic arithmetic operations (add, sub, mul, div, matmul, exp, log, etc.)
//! - **`transforms/`** - Shape transformations (reshape, permute, transpose, cat, split)
//! - **`indexing/`** - Indexing operations (select, gather, index_select, masked_fill)
//! - **`reductions/`** - Reduction operations (sum, mean, min, max, std, var, norm)
//! - **`utils/`** - Common gradient computation utilities
//!
//! # Performance Benefits
//!
//! - **Zero-cost dispatch**: Enum-based gradient functions eliminate virtual function overhead
//! - **Compile-time optimization**: All gradient paths are known at compile time
//! - **Memory efficiency**: Minimal overhead for gradient tracking metadata
//! - **SIMD optimization**: Gradient computations leverage vectorized operations
//!
//! # Thread Safety
//!
//! All gradient functions are thread-safe and can be used concurrently across multiple threads.
//! The system uses thread-local storage for gradient state management, ensuring proper isolation.
//!
//! # Integration with GradTrack
//!
//! Gradient functions integrate seamlessly with the GradTrack system:
//! - **Automatic registration**: Tensor operations automatically create appropriate gradient functions
//! - **Computation graph**: Gradient functions form the nodes of the automatic differentiation graph
//! - **Backward pass**: The `apply()` method computes gradients during backward propagation
//! - **Memory management**: Efficient gradient accumulation and cleanup

use crate::tensor::core::Tensor;

pub mod indexing;
pub mod ops;
pub mod reductions;
pub mod transforms;
pub mod utils;

/// Enumeration of gradient functions for all supported tensor operations
///
/// This enum provides zero-cost dispatch for gradient computation during backward passes.
/// Each variant contains the metadata and saved tensors needed to compute gradients
/// for its corresponding forward operation. The enum-based approach eliminates virtual
/// function overhead while maintaining type safety and enabling compile-time optimizations.
///
/// # Variants
///
/// ## Arithmetic Operations
/// * `Add` - Addition operations (tensor + tensor, tensor + scalar)
/// * `Sub` - Subtraction operations (tensor - tensor, tensor - scalar)
/// * `Mul` - Multiplication operations (tensor * tensor, tensor * scalar)
/// * `Div` - Division operations (tensor / tensor, tensor / scalar)
/// * `MatMul` - Matrix multiplication operations
///
/// ## Mathematical Functions
/// * `Exp` - Exponential function (e^x)
/// * `Log` - Natural logarithm function (ln(x))
/// * `Sqrt` - Square root function (√x)
/// * `PowScalar` - Power function with scalar exponent (x^n)
/// * `PowTensor` - Power function with tensor exponent (x^y)
///
/// ## Activation Functions
/// * `Relu` - Rectified Linear Unit activation
/// * `LeakyRelu` - Leaky ReLU activation with configurable slope
/// * `Tanh` - Hyperbolic tangent activation
/// * `Sigmoid` - Sigmoid activation function
/// * `Softmax` - Softmax activation with dimension specification
///
/// ## Shape Transformations
/// * `Reshape` - Shape transformation while preserving memory layout
/// * `Contiguous` - Identity operation for gradient tracking
/// * `Permute` - Dimension reordering with permutation specification
/// * `Transpose` - Dimension swapping (2D matrix transpose)
/// * `SliceView` - Slice view with start, step, and length parameters
///
/// ## Concatenation and Splitting
/// * `Cat` - Tensor concatenation along specified dimension
/// * `Split` - Tensor splitting with dimension and size parameters
///
/// ## Reduction Operations
/// * `ReduceSum` - Sum reduction over all elements
/// * `ReduceMean` - Mean reduction over all elements
/// * `ReduceMin` - Minimum reduction over all elements
/// * `ReduceMax` - Maximum reduction over all elements
/// * `ReduceStd` - Standard deviation reduction (population)
/// * `ReduceVar` - Variance reduction (population)
/// * `ReduceNorm` - L2 norm reduction
/// * `ReduceSumDims` - Sum reduction over specific dimensions
/// * `ReduceMeanDims` - Mean reduction over specific dimensions
/// * `ReduceMinDims` - Minimum reduction over specific dimensions
/// * `ReduceMaxDims` - Maximum reduction over specific dimensions
/// * `ReduceStdDims` - Standard deviation reduction over specific dimensions
/// * `ReduceVarDims` - Variance reduction over specific dimensions
/// * `ReduceNormDims` - L2 norm reduction over specific dimensions
///
/// ## Indexing Operations
/// * `IndexSelect` - Index selection along specified dimension
/// * `Gather` - Gather operation with index tensor
/// * `MaskedFill` - Masked fill operation with boolean mask
/// * `Select` - Element selection along specified dimension
///
/// ## View Operations
/// * `ElementView` - Single element view for gradient accumulation
/// * `ElementCollection` - Collection of element views back into tensor
///
/// ## Special Cases
/// * `None` - Leaf tensor with no gradient function (input parameter)
///
/// # Performance Characteristics
///
/// - **Zero-cost dispatch**: No virtual function overhead
/// - **Memory efficiency**: Minimal metadata storage per operation
/// - **Compile-time optimization**: All gradient paths known at compile time
/// - **SIMD support**: Gradient computations leverage vectorized operations
///
/// # Thread Safety
///
/// All gradient function variants are thread-safe and can be used concurrently.
/// The system uses thread-local storage for gradient state management.
#[derive(Debug, Clone)]
pub enum GradFn {
    /// Addition gradient function (tensor + tensor or tensor + scalar)
    Add {
        is_tensor_add: bool,
        /// Original shapes for broadcasting (first input, second input)
        original_shapes: Option<(Vec<usize>, Vec<usize>)>,
    },
    /// Subtraction gradient function (tensor - tensor or tensor - scalar)
    Sub {
        is_tensor_sub: bool,
        /// Original shapes for broadcasting (first input, second input)
        original_shapes: Option<(Vec<usize>, Vec<usize>)>,
    },
    /// Multiplication gradient function (tensor * tensor or tensor * scalar)
    Mul {
        is_tensor_mul: bool,
        scalar: Option<f32>,
        operands: Option<Vec<Tensor>>,
        /// Original shapes for broadcasting (first input, second input)
        original_shapes: Option<(Vec<usize>, Vec<usize>)>,
    },
    /// Division gradient function (tensor / tensor or tensor / scalar)
    Div {
        is_tensor_div: bool,
        scalar: Option<f32>,
        operands: Option<Vec<Tensor>>,
        /// Original shapes for broadcasting (first input, second input)
        original_shapes: Option<(Vec<usize>, Vec<usize>)>,
    },
    /// Matrix multiplication gradient function
    MatMul {
        /// Left operand for gradient computation
        left_operand: Box<Tensor>,
        /// Right operand for gradient computation
        right_operand: Box<Tensor>,
        /// Which operands require gradients (left, right)
        requires_grad: (bool, bool),
    },
    /// Reshape gradient function
    Reshape {
        /// Original shape before reshape for gradient restoration
        original_shape: Vec<usize>,
    },
    /// Contiguous gradient function (identity operation for gradients)
    Contiguous {
        /// Input shape (same as output shape for contiguous operation)
        input_shape: Vec<usize>,
    },
    /// Exponential gradient function
    Exp {
        /// Saved forward output for efficient grad computation (d/dx exp(x) = exp(x))
        saved_output: Box<Tensor>,
    },
    /// Tanh gradient function (saves output)
    Tanh { saved_output: Box<Tensor> },
    /// Sigmoid gradient function (saves output)
    Sigmoid { saved_output: Box<Tensor> },
    /// Natural logarithm gradient function
    Log {
        /// Saved input for efficient grad computation (d/dx log(x) = 1/x)
        saved_input: Box<Tensor>,
    },
    /// Softmax gradient function
    Softmax {
        dim: usize,
        saved_output: Box<Tensor>,
    },
    /// Power with scalar exponent gradient function
    PowScalar {
        exponent: f32,
        saved_input: Box<Tensor>,
    },
    /// Power with tensor exponent gradient function
    PowTensor {
        saved_base: Box<Tensor>,
        saved_exponent: Box<Tensor>,
    },
    /// Concatenation gradient function
    Cat {
        /// Concatenation dimension
        dim: usize,
        /// Sizes along concatenation dim for inputs that require grad (in registration order)
        input_sizes: Vec<usize>,
        /// Full shapes for inputs that require grad (to allocate grads)
        input_shapes: Vec<Vec<usize>>,
    },
    /// Split gradient function (for each split output)
    Split {
        /// Dimension split along
        dim: usize,
        /// Start index within the split dimension in the original input
        start: usize,
        /// Length of this split along the split dimension
        length: usize,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Square root gradient function
    Sqrt {
        /// Saved forward output for efficient grad computation (d/dx sqrt(x) = 0.5 / sqrt(x))
        saved_output: Box<Tensor>,
    },
    /// ReLU gradient function
    Relu { saved_input: Box<Tensor> },
    /// LeakyReLU gradient function
    LeakyRelu {
        negative_slope: f32,
        saved_input: Box<Tensor>,
    },
    /// Permute gradient function
    Permute {
        /// Permutation applied in forward (dest axis i comes from src axis dims[i])
        dims: Vec<usize>,
        /// Original input shape to restore on backward
        input_shape: Vec<usize>,
    },
    /// Transpose gradient function
    Transpose {
        /// Dimensions that were swapped in forward
        dim0: usize,
        dim1: usize,
        /// Original input shape to restore on backward
        input_shape: Vec<usize>,
    },
    /// Slice view gradient function
    SliceView {
        /// Starting index of the slice
        start: usize,
        /// Step size of the slice (1 for contiguous, >1 for strided)
        step: usize,
        /// Length of the slice
        length: usize,
        /// Original input shape to restore on backward
        input_shape: Vec<usize>,
    },
    /// Reduction: sum over all elements
    ReduceSum {
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Reduction: mean over all elements
    ReduceMean {
        /// Original input shape
        input_shape: Vec<usize>,
        /// Number of elements in input for scaling the gradient
        numel: usize,
    },
    /// Reduction: min over all elements
    ReduceMin {
        /// Saved forward min value (shape [1])
        saved_output: Box<Tensor>,
        /// Saved forward input to compute mask on backward
        saved_input: Box<Tensor>,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Reduction: max over all elements
    ReduceMax {
        /// Saved forward max value (shape [1])
        saved_output: Box<Tensor>,
        /// Saved forward input to compute mask on backward
        saved_input: Box<Tensor>,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Reduction: min over specific dimensions
    ReduceMinDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Saved forward min values (shape of output)
        saved_output: Box<Tensor>,
        /// Saved forward input to compute mask on backward
        saved_input: Box<Tensor>,
    },
    /// Reduction: max over specific dimensions
    ReduceMaxDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Saved forward max values (shape of output)
        saved_output: Box<Tensor>,
        /// Saved forward input to compute mask on backward
        saved_input: Box<Tensor>,
    },
    /// Reduction: sum over specific dimensions
    ReduceSumDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
    },
    /// Reduction: mean over specific dimensions
    ReduceMeanDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
    },
    /// Reduction: std over all elements (population)
    ReduceStd {
        saved_mean: Box<Tensor>,
        saved_std: Box<Tensor>,
        saved_input: Box<Tensor>,
        input_shape: Vec<usize>,
    },
    /// Reduction: std over specific dimensions (population)
    ReduceStdDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Saved mean per output position
        saved_mean: Box<Tensor>,
        /// Saved std per output position
        saved_std: Box<Tensor>,
        /// Saved input
        saved_input: Box<Tensor>,
    },
    /// Reduction: L2 norm over all elements
    ReduceNorm {
        saved_norm: Box<Tensor>,
        saved_input: Box<Tensor>,
        input_shape: Vec<usize>,
    },
    /// Reduction: L2 norm over dims
    ReduceNormDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Saved norm values
        saved_norm: Box<Tensor>,
        /// Saved input
        saved_input: Box<Tensor>,
    },
    /// Reduction: var over all elements (population)
    ReduceVar {
        saved_mean: Box<Tensor>,
        saved_input: Box<Tensor>,
        input_shape: Vec<usize>,
    },
    /// Reduction: var over specific dimensions (population)
    ReduceVarDims {
        /// Reduced dimensions (sorted, unique)
        dims: Vec<usize>,
        /// Whether forward kept reduced dims as size-1
        keepdim: bool,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Saved mean per output position
        saved_mean: Box<Tensor>,
        /// Saved input
        saved_input: Box<Tensor>,
    },
    /// Index select gradient function
    IndexSelect {
        /// Dimension selected along
        dim: usize,
        /// Indices used in forward
        indices: Vec<usize>,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Gather gradient function
    Gather {
        /// Dimension gathered along
        dim: usize,
        /// Flattened indices used in forward (length = product of index_shape)
        indices: Vec<usize>,
        /// Original input shape
        input_shape: Vec<usize>,
        /// Index/output shape
        index_shape: Vec<usize>,
    },
    /// Masked fill gradient function
    MaskedFill {
        /// Boolean mask flattened (length = numel)
        mask: Vec<bool>,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Select gradient function
    Select {
        /// Dimension selected
        dim: usize,
        /// Fixed index in that dimension
        index: usize,
        /// Original input shape
        input_shape: Vec<usize>,
    },
    /// Element view gradient function (single element from tensor)
    ElementView {
        /// Source tensor ID
        source_id: usize,
        /// Index of the element in the source tensor
        element_index: usize,
        /// Shape of the source tensor for gradient accumulation
        source_shape: Vec<usize>,
    },
    /// Collection of element views back into tensor
    ElementCollection {
        /// IDs of the element view tensors
        element_ids: Vec<usize>,
        /// Shape of the resulting collected tensor
        result_shape: Vec<usize>,
    },
    /// Leaf tensor (no gradient function)
    None,
}

impl GradFn {
    /// Apply the gradient function during backward pass computation
    ///
    /// This method computes gradients for the inputs to the forward operation
    /// using the chain rule of differentiation. The gradient computation is
    /// specific to each operation type and uses saved forward pass data for
    /// efficient computation.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - Gradient of the output with respect to the loss
    ///
    /// # Returns
    ///
    /// Vector of gradients for the operation inputs. Each element corresponds
    /// to an input tensor in the order they were provided to the forward operation.
    /// `None` indicates that the corresponding input does not require gradients.
    ///
    /// # Gradient Computation
    ///
    /// The method performs operation-specific gradient computation:
    /// - **Arithmetic operations**: Apply chain rule with operation derivatives
    /// - **Mathematical functions**: Use saved forward outputs for efficient computation
    /// - **Shape transformations**: Restore original shapes for gradient accumulation
    /// - **Reduction operations**: Expand gradients to original tensor shapes
    /// - **Indexing operations**: Scatter gradients back to original positions
    ///
    /// # Performance
    ///
    /// - **Efficient computation**: Uses saved forward pass data to avoid recomputation
    /// - **SIMD optimization**: Leverages vectorized operations where possible
    /// - **Memory efficiency**: Minimizes temporary allocations during gradient computation
    /// - **Zero-cost dispatch**: Enum-based dispatch eliminates virtual function overhead
    ///
    /// # Thread Safety
    ///
    /// This method is thread-safe and can be called concurrently across multiple threads.
    /// Each gradient computation is independent and uses thread-local storage.
    pub fn apply(&self, grad_output: &Tensor) -> Vec<Option<Tensor>> {
        match self {
            GradFn::Add {
                is_tensor_add,
                original_shapes,
            } => ops::add::apply_add(*is_tensor_add, original_shapes.as_ref(), grad_output),
            GradFn::Sub {
                is_tensor_sub,
                original_shapes,
            } => ops::sub::apply_sub(*is_tensor_sub, original_shapes.as_ref(), grad_output),
            GradFn::Mul {
                is_tensor_mul,
                scalar,
                operands,
                original_shapes,
            } => ops::mul::apply_mul(
                *is_tensor_mul,
                *scalar,
                operands.as_ref(),
                original_shapes.as_ref(),
                grad_output,
            ),
            GradFn::Div {
                is_tensor_div,
                scalar,
                operands,
                original_shapes,
            } => ops::div::apply_div(
                *is_tensor_div,
                *scalar,
                operands.as_ref(),
                original_shapes.as_ref(),
                grad_output,
            ),
            GradFn::MatMul {
                left_operand,
                right_operand,
                requires_grad,
            } => {
                ops::matmul::apply_matmul(left_operand, right_operand, *requires_grad, grad_output)
            }
            GradFn::Reshape { original_shape } => {
                transforms::reshape::apply_reshape(original_shape, grad_output)
            }
            GradFn::Contiguous { input_shape: _ } => {
                // Contiguous is an identity operation for gradients - just pass through
                vec![Some(grad_output.clone())]
            }
            GradFn::Exp { saved_output } => ops::exp::apply_exp(saved_output, grad_output),
            GradFn::Tanh { saved_output } => ops::tanh::apply_tanh(saved_output, grad_output),
            GradFn::Sigmoid { saved_output } => {
                ops::sigmoid::apply_sigmoid(saved_output, grad_output)
            }
            GradFn::Log { saved_input } => ops::log::apply_log(saved_input, grad_output),
            GradFn::Softmax { dim, saved_output } => {
                ops::softmax::apply_softmax(*dim, saved_output, grad_output)
            }
            GradFn::PowScalar {
                exponent,
                saved_input,
            } => ops::pow::apply_pow_scalar(*exponent, saved_input, grad_output),
            GradFn::PowTensor {
                saved_base,
                saved_exponent,
            } => ops::pow::apply_pow_tensor(saved_base, saved_exponent, grad_output),
            GradFn::Cat {
                dim,
                input_sizes,
                input_shapes,
            } => transforms::cat::apply_cat(*dim, input_sizes, input_shapes, grad_output),
            GradFn::Split {
                dim,
                start,
                length,
                input_shape,
            } => transforms::split::apply_split(*dim, *start, *length, input_shape, grad_output),
            GradFn::Sqrt { saved_output } => ops::sqrt::apply_sqrt(saved_output, grad_output),
            GradFn::Relu { saved_input } => ops::relu::apply_relu(saved_input, grad_output),
            GradFn::LeakyRelu {
                negative_slope,
                saved_input,
            } => ops::leaky_relu::apply_leaky_relu(*negative_slope, saved_input, grad_output),
            GradFn::Permute { dims, input_shape } => {
                transforms::permute::apply_permute(dims, input_shape, grad_output)
            }
            GradFn::Transpose {
                dim0,
                dim1,
                input_shape,
            } => transforms::transpose::apply_transpose(*dim0, *dim1, input_shape, grad_output),
            GradFn::SliceView {
                start,
                step,
                length,
                input_shape,
            } => transforms::slice_view::apply_slice_view(
                *start,
                *step,
                *length,
                input_shape,
                grad_output,
            ),
            GradFn::ReduceSum { input_shape } => {
                reductions::sum::apply_reduce_sum(input_shape, grad_output)
            }
            GradFn::ReduceMean { input_shape, numel } => {
                reductions::mean::apply_reduce_mean(input_shape, *numel, grad_output)
            }
            GradFn::ReduceMin {
                saved_output,
                saved_input,
                input_shape,
            } => reductions::min::apply_reduce_min(
                input_shape,
                saved_output,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceMax {
                saved_output,
                saved_input,
                input_shape,
            } => reductions::max::apply_reduce_max(
                input_shape,
                saved_output,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceMinDims {
                dims,
                keepdim,
                input_shape,
                saved_output,
                saved_input,
            } => reductions::min::apply_reduce_min_dims(
                dims,
                *keepdim,
                input_shape,
                saved_output,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceMaxDims {
                dims,
                keepdim,
                input_shape,
                saved_output,
                saved_input,
            } => reductions::max::apply_reduce_max_dims(
                dims,
                *keepdim,
                input_shape,
                saved_output,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceNorm {
                saved_norm,
                saved_input,
                input_shape,
            } => reductions::norm::apply_reduce_norm(
                input_shape,
                saved_norm,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceNormDims {
                dims,
                keepdim,
                input_shape,
                saved_norm,
                saved_input,
            } => reductions::norm::apply_reduce_norm_dims(
                dims,
                *keepdim,
                input_shape,
                saved_norm,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceStd {
                saved_mean,
                saved_std,
                saved_input,
                input_shape,
            } => reductions::std::apply_reduce_std(
                input_shape,
                saved_mean,
                saved_std,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceStdDims {
                dims,
                keepdim,
                input_shape,
                saved_mean,
                saved_std,
                saved_input,
            } => reductions::std::apply_reduce_std_dims(
                dims,
                *keepdim,
                input_shape,
                saved_mean,
                saved_std,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceVar {
                saved_mean,
                saved_input,
                input_shape,
            } => {
                reductions::var::apply_reduce_var(input_shape, saved_mean, saved_input, grad_output)
            }
            GradFn::ReduceVarDims {
                dims,
                keepdim,
                input_shape,
                saved_mean,
                saved_input,
            } => reductions::var::apply_reduce_var_dims(
                dims,
                *keepdim,
                input_shape,
                saved_mean,
                saved_input,
                grad_output,
            ),
            GradFn::ReduceSumDims {
                dims,
                input_shape,
                keepdim,
            } => reductions::sum::apply_reduce_sum_dims(dims, input_shape, *keepdim, grad_output),
            GradFn::ReduceMeanDims {
                dims,
                input_shape,
                keepdim,
            } => reductions::mean::apply_reduce_mean_dims(dims, input_shape, *keepdim, grad_output),
            GradFn::IndexSelect {
                dim,
                indices,
                input_shape,
            } => {
                indexing::index_select::apply_index_select(*dim, indices, input_shape, grad_output)
            }
            GradFn::Gather {
                dim,
                indices,
                input_shape,
                index_shape,
            } => {
                indexing::gather::apply_gather(*dim, indices, input_shape, index_shape, grad_output)
            }
            GradFn::MaskedFill { mask, input_shape } => {
                indexing::masked_fill::apply_masked_fill(mask, input_shape, grad_output)
            }
            GradFn::Select {
                dim,
                index,
                input_shape,
            } => indexing::select::apply_select(*dim, *index, input_shape, grad_output),
            GradFn::ElementView {
                source_id: _,
                element_index: _,
                source_shape: _,
            } => {
                // Element view: Currently working with scalar gradient accumulation
                // The gradient engine will accumulate this into the source tensor
                // For now, return the gradient as-is since basic flow is working
                if grad_output.size() == 1 {
                    vec![Some(grad_output.clone())]
                } else {
                    // If grad_output is not scalar, sum it to create a scalar gradient
                    let scalar_grad = grad_output.sum();
                    vec![Some(scalar_grad)]
                }
            }
            GradFn::ElementCollection {
                element_ids,
                result_shape: _,
            } => {
                // Distribute gradients back to element views
                // Each element gets a slice of the gradient output
                let mut gradients = Vec::new();
                for (i, _id) in element_ids.iter().enumerate() {
                    if i < grad_output.size() {
                        // Create gradient for this element
                        let grad_value = grad_output.data()[i];
                        let elem_grad = Tensor::from_slice(&[grad_value], vec![1]).unwrap();
                        gradients.push(Some(elem_grad));
                    } else {
                        gradients.push(None);
                    }
                }
                gradients
            }
            GradFn::None => Vec::new(),
        }
    }

    /// Get the human-readable name of the gradient function for debugging and logging
    ///
    /// This method returns a string identifier for the gradient function type,
    /// which is useful for debugging, logging, and understanding the computation
    /// graph structure during automatic differentiation.
    ///
    /// # Returns
    ///
    /// A static string slice containing the name of the gradient function variant
    ///
    /// # Usage
    ///
    /// The returned name can be used for:
    /// - **Debugging**: Identifying gradient function types in error messages
    /// - **Logging**: Tracking gradient computation during training
    /// - **Profiling**: Analyzing performance of different gradient operations
    /// - **Visualization**: Displaying computation graph structure
    ///
    /// # Performance
    ///
    /// - **Constant time**: O(1) time complexity for name retrieval
    /// - **No allocation**: Returns static string slices
    /// - **Zero overhead**: Compile-time string resolution
    pub fn name(&self) -> &'static str {
        match self {
            GradFn::Add { .. } => "Add",
            GradFn::Sub { .. } => "Sub",
            GradFn::Mul { .. } => "Mul",
            GradFn::Div { .. } => "Div",
            GradFn::MatMul { .. } => "MatMul",
            GradFn::Reshape { .. } => "Reshape",
            GradFn::Contiguous { .. } => "Contiguous",
            GradFn::Exp { .. } => "Exp",
            GradFn::Log { .. } => "Log",
            GradFn::Cat { .. } => "Cat",
            GradFn::Split { .. } => "Split",
            GradFn::Sqrt { .. } => "Sqrt",
            GradFn::Permute { .. } => "Permute",
            GradFn::Transpose { .. } => "Transpose",
            GradFn::SliceView { .. } => "SliceView",
            GradFn::ReduceSum { .. } => "ReduceSum",
            GradFn::ReduceMean { .. } => "ReduceMean",
            GradFn::ReduceMin { .. } => "ReduceMin",
            GradFn::ReduceMax { .. } => "ReduceMax",
            GradFn::ReduceMinDims { .. } => "ReduceMinDims",
            GradFn::ReduceMaxDims { .. } => "ReduceMaxDims",
            GradFn::ReduceSumDims { .. } => "ReduceSumDims",
            GradFn::ReduceMeanDims { .. } => "ReduceMeanDims",
            GradFn::ReduceStd { .. } => "ReduceStd",
            GradFn::ReduceStdDims { .. } => "ReduceStdDims",
            GradFn::ReduceNorm { .. } => "ReduceNorm",
            GradFn::ReduceNormDims { .. } => "ReduceNormDims",
            GradFn::ReduceVar { .. } => "ReduceVar",
            GradFn::ReduceVarDims { .. } => "ReduceVarDims",
            GradFn::PowScalar { .. } => "PowScalar",
            GradFn::PowTensor { .. } => "PowTensor",
            GradFn::Softmax { .. } => "Softmax",
            GradFn::Relu { .. } => "Relu",
            GradFn::Tanh { .. } => "Tanh",
            GradFn::Sigmoid { .. } => "Sigmoid",
            GradFn::LeakyRelu { .. } => "LeakyRelu",
            GradFn::IndexSelect { .. } => "IndexSelect",
            GradFn::Gather { .. } => "Gather",
            GradFn::MaskedFill { .. } => "MaskedFill",
            GradFn::Select { .. } => "Select",
            GradFn::ElementView { .. } => "ElementView",
            GradFn::ElementCollection { .. } => "ElementCollection",
            GradFn::None => "Leaf",
        }
    }
}
