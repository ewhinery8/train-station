use super::super::utils::reduce_matmul_grad_to_operand_shape;
use crate::tensor::ops::matmul::dispatch::MatMulKernels;
use crate::Tensor;

/// Apply gradient computation for matrix multiplication operations
///
/// This function computes gradients for all supported matmul operation types:
/// - 1D @ 1D: Dot product gradients
/// - 1D @ 2D: Vector-matrix multiplication gradients  
/// - 2D @ 1D: Matrix-vector multiplication gradients
/// - 2D @ 2D: Matrix-matrix multiplication gradients
/// - ND @ ND: Batched matrix multiplication gradients with broadcasting
///
/// The implementation leverages existing matmul kernels for efficient gradient computation
/// and handles shape validation and broadcasting correctly.
///
/// # Arguments
/// * `left_operand` - Left operand tensor used in forward pass
/// * `right_operand` - Right operand tensor used in forward pass  
/// * `requires_grad` - Tuple indicating which operands require gradients (left, right)
/// * `grad_output` - Gradient flowing back from the output
///
/// # Returns
/// Vector containing gradients for left and right operands (None if gradient not required)
pub(crate) fn apply_matmul(
    left_operand: &Tensor,
    right_operand: &Tensor,
    requires_grad: (bool, bool),
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let left_shape = left_operand.shape().dims();
    let right_shape = right_operand.shape().dims();
    let op_type = MatMulKernels::classify_operation(left_shape, right_shape);

    let mut gradients = Vec::new();

    // Compute left gradient if required
    let grad_left = if requires_grad.0 {
        Some(compute_left_gradient(
            left_operand,
            right_operand,
            grad_output,
            &op_type,
        ))
    } else {
        None
    };

    // Compute right gradient if required
    let grad_right = if requires_grad.1 {
        Some(compute_right_gradient(
            left_operand,
            right_operand,
            grad_output,
            &op_type,
        ))
    } else {
        None
    };

    gradients.push(grad_left);
    gradients.push(grad_right);

    gradients
}

/// Compute gradient for left operand
fn compute_left_gradient(
    left_operand: &Tensor,
    right_operand: &Tensor,
    grad_output: &Tensor,
    op_type: &crate::tensor::ops::matmul::dispatch::MatMulOpType,
) -> Tensor {
    use crate::tensor::ops::matmul::dispatch::MatMulOpType;

    let grad = match op_type {
        MatMulOpType::Dot1D1D => {
            // 1D @ 1D: grad_a = grad_out * b (element-wise multiply)
            // grad_out is scalar, b is vector -> broadcast multiply
            let grad_out_scalar = grad_output.value(); // Extract scalar value
            right_operand.mul_scalar(grad_out_scalar)
        }
        MatMulOpType::Vec1D2D => {
            // 1D @ 2D: grad_a = grad_out @ b.T
            // grad_out: [n], b: [k, n] -> grad_a: [k]
            let b_transposed = right_operand.transpose(0, 1);
            // Always make transposed tensor contiguous to ensure correct computation
            let b_transposed_contiguous = b_transposed.contiguous();
            crate::gradtrack::with_no_grad(|| grad_output.matmul(&b_transposed_contiguous))
        }
        MatMulOpType::Mat2D1D => {
            // 2D @ 1D: grad_a = grad_out.outer(b) = grad_out @ b.T
            // grad_out: [m], b: [k] -> grad_a: [m, k]
            // Reshape grad_out to [m, 1] and b to [1, k] for matmul
            let grad_reshaped = grad_output.view(vec![grad_output.size() as i32, 1]);
            let b_reshaped = right_operand.view(vec![1, right_operand.size() as i32]);
            crate::gradtrack::with_no_grad(|| grad_reshaped.matmul(&b_reshaped))
        }
        MatMulOpType::Mat2D2D => {
            // 2D @ 2D: grad_a = grad_out @ b.T
            let b_transposed = right_operand.transpose(0, 1);
            // Always make transposed tensor contiguous to ensure correct computation
            let b_transposed_contiguous = b_transposed.contiguous();
            crate::gradtrack::with_no_grad(|| grad_output.matmul(&b_transposed_contiguous))
        }
        MatMulOpType::BatchedND => {
            // Generalized ND gradient using unified classifier and reshapes
            compute_batched_left_gradient_general(left_operand, right_operand, grad_output)
        }
    };

    // Reduce broadcasted batch dims to match the left operand shape using
    // matmul-aware reduction with correct keep_last (1 for vector-like, 2 for matrix-like)
    let left_shape = left_operand.shape().dims();
    let right_shape = right_operand.shape().dims();
    let (l_mat_dims, _r_mat_dims) = crate::Tensor::classify_nd_matmul_dims(left_shape, right_shape);
    reduce_matmul_grad_to_operand_shape(&grad, left_shape, l_mat_dims)
}

/// Compute gradient for right operand
fn compute_right_gradient(
    left_operand: &Tensor,
    right_operand: &Tensor,
    grad_output: &Tensor,
    op_type: &crate::tensor::ops::matmul::dispatch::MatMulOpType,
) -> Tensor {
    use crate::tensor::ops::matmul::dispatch::MatMulOpType;

    let grad = match op_type {
        MatMulOpType::Dot1D1D => {
            // 1D @ 1D: grad_b = grad_out * a (element-wise multiply)
            // grad_out is scalar, a is vector -> broadcast multiply
            let grad_out_scalar = grad_output.value(); // Extract scalar value
            left_operand.mul_scalar_optimized(grad_out_scalar)
        }
        MatMulOpType::Vec1D2D => {
            // 1D @ 2D: grad_b = a.outer(grad_out) = a.T @ grad_out
            // a: [k], grad_out: [n] -> grad_b: [k, n]
            // Reshape a to [k, 1] and grad_out to [1, n] for matmul
            let a_reshaped = left_operand.view(vec![left_operand.size() as i32, 1]);
            let grad_reshaped = grad_output.view(vec![1, grad_output.size() as i32]);
            crate::gradtrack::with_no_grad(|| a_reshaped.matmul(&grad_reshaped))
        }
        MatMulOpType::Mat2D1D => {
            // 2D @ 1D: grad_b = a.T @ grad_out
            let a_transposed = left_operand.transpose(0, 1);
            // Always make transposed tensor contiguous to ensure correct computation
            let a_transposed_contiguous = a_transposed.contiguous();
            crate::gradtrack::with_no_grad(|| a_transposed_contiguous.matmul(grad_output))
        }
        MatMulOpType::Mat2D2D => {
            // 2D @ 2D: grad_b = a.T @ grad_out
            let a_transposed = left_operand.transpose(0, 1);
            // Always make transposed tensor contiguous to ensure correct computation
            let a_transposed_contiguous = a_transposed.contiguous();
            crate::gradtrack::with_no_grad(|| a_transposed_contiguous.matmul(grad_output))
        }
        MatMulOpType::BatchedND => {
            // Generalized ND gradient using unified classifier and reshapes
            compute_batched_right_gradient_general(left_operand, right_operand, grad_output)
        }
    };

    // Matmul-aware reduction with correct keep_last (1 for vector-like, 2 for matrix-like)
    let right_shape = right_operand.shape().dims();
    let left_shape = left_operand.shape().dims();
    let (_l_mat_dims, r_mat_dims) = crate::Tensor::classify_nd_matmul_dims(left_shape, right_shape);
    reduce_matmul_grad_to_operand_shape(&grad, right_shape, r_mat_dims)
}

fn compute_batched_left_gradient_general(
    left_operand: &Tensor,
    right_operand: &Tensor,
    grad_output: &Tensor,
) -> Tensor {
    let left_shape = left_operand.shape().dims();
    let right_shape = right_operand.shape().dims();

    let (l_md, r_md) = crate::Tensor::classify_nd_matmul_dims(left_shape, right_shape);

    crate::gradtrack::with_no_grad(|| {
        // Build B^T or B-view so that matmul(go_use, b_use) computes raw left grad
        let b_use = if r_md == 2 {
            let rr = right_shape.len();
            right_operand.transpose(rr - 2, rr - 1).contiguous()
        } else {
            // r_md == 1: view as [..., 1, K]
            let mut dims: Vec<i32> = right_shape.iter().map(|&d| d as i32).collect();
            dims.push(1); // [..., K, 1] but we need [..., 1, K]
                          // Build [...batch..., 1, K]
            let mut view_dims = Vec::with_capacity(dims.len());
            if dims.len() == 1 {
                // Right is [K]
                view_dims.push(1);
                view_dims.push(dims[0]);
            } else {
                // [B..., K] -> [B..., 1, K]
                view_dims.extend_from_slice(
                    &right_shape[..right_shape.len() - 1]
                        .iter()
                        .map(|&d| d as i32)
                        .collect::<Vec<i32>>(),
                );
                view_dims.push(1);
                view_dims.push(*right_shape.last().unwrap() as i32);
            }
            right_operand.view(view_dims)
        };

        // Prepare grad_output to appropriate 2D form when needed
        let go_use = if l_md == 2 {
            if r_md == 1 {
                // [..., M] -> [..., M, 1]
                let mut dims: Vec<i32> = grad_output
                    .shape()
                    .dims()
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                dims.push(1);
                grad_output.view(dims)
            } else {
                grad_output.clone()
            }
        } else {
            // l_md == 1
            if r_md == 2 {
                // [..., N] -> [..., 1, N]
                let go_dims = grad_output.shape().dims();
                if go_dims.is_empty() {
                    grad_output.view(vec![1, 1])
                } else {
                    let mut dims: Vec<i32> = go_dims[..go_dims.len() - 1]
                        .iter()
                        .map(|&d| d as i32)
                        .collect();
                    dims.push(1);
                    dims.push(*go_dims.last().unwrap() as i32);
                    grad_output.view(dims)
                }
            } else {
                // r_md == 1: [...batch...] -> [...batch..., 1, 1]
                let mut dims: Vec<i32> = grad_output
                    .shape()
                    .dims()
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                dims.push(1);
                dims.push(1);
                grad_output.view(dims)
            }
        };

        go_use.matmul(&b_use)
    })
}

/// Compute batched right gradient for mixed dimensionality cases
fn compute_batched_right_gradient_general(
    left_operand: &Tensor,
    right_operand: &Tensor,
    grad_output: &Tensor,
) -> Tensor {
    let left_shape = left_operand.shape().dims();
    let right_shape = right_operand.shape().dims();

    let (l_md, r_md) = crate::Tensor::classify_nd_matmul_dims(left_shape, right_shape);

    crate::gradtrack::with_no_grad(|| {
        // Build A^T or A-view so that matmul(a_use, go_use) computes raw right grad
        let a_use = if l_md == 2 {
            let lr = left_shape.len();
            left_operand.transpose(lr - 2, lr - 1).contiguous()
        } else {
            // l_md == 1: [..., K] -> [..., K, 1]
            let mut dims: Vec<i32> = left_shape.iter().map(|&d| d as i32).collect();
            dims.push(1);
            left_operand.view(dims)
        };

        // Prepare grad_output appropriately
        let go_use = if r_md == 2 {
            if l_md == 1 {
                // [..., N] -> [..., 1, N]
                let go_dims = grad_output.shape().dims();
                let mut dims: Vec<i32> = go_dims[..go_dims.len() - 1]
                    .iter()
                    .map(|&d| d as i32)
                    .collect();
                dims.push(1);
                dims.push(*go_dims.last().unwrap() as i32);
                grad_output.view(dims)
            } else {
                grad_output.clone()
            }
        } else {
            // r_md == 1: need [..., M, 1]
            let go_dims = grad_output.shape().dims();
            if go_dims.is_empty() {
                grad_output.view(vec![1, 1])
            } else {
                let mut dims: Vec<i32> = go_dims.iter().map(|&d| d as i32).collect();
                dims.push(1);
                grad_output.view(dims)
            }
        };

        let mut raw = a_use.matmul(&go_use);
        // Ensure the last `keep_last` dims match the operand's matrix dims order.
        // For r_md == 1, raw has trailing dims [K, 1]; swap to [1, K] so keep_last=1 refers to K.
        if r_md == 1 {
            let rr = raw.shape().dims().len();
            if rr >= 2 {
                raw = raw.transpose(rr - 2, rr - 1).contiguous();
            }
        }
        raw
    })
}
