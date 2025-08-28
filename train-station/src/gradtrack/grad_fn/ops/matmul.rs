use crate::tensor::core::Tensor;

pub(crate) fn apply_matmul(
    left_operand: &Tensor,
    right_operand: &Tensor,
    requires_grad: (bool, bool),
    grad_output: &Tensor,
) -> Vec<Option<Tensor>> {
    let left_shape = left_operand.shape();
    let right_shape = right_operand.shape();
    let _grad_shape = grad_output.shape();

    // Handle different matmul patterns
    match (left_shape.rank(), right_shape.rank()) {
        // 1D @ 1D (dot product): a • b = scalar
        // grad_a = grad_output * b, grad_b = grad_output * a
        (1, 1) => {
            let grad_left = if requires_grad.0 {
                Some(grad_output.mul_tensor(right_operand))
            } else {
                None
            };
            let grad_right = if requires_grad.1 {
                Some(grad_output.mul_tensor(left_operand))
            } else {
                None
            };
            vec![grad_left, grad_right]
        }

        // 1D @ 2D (vector @ matrix): v @ M = result_vector
        // grad_v = grad_output @ M^T, grad_M = v^T @ grad_output (but v is 1D, so outer product)
        (1, 2) => {
            let grad_left = if requires_grad.0 {
                // Manually compute grad_output @ M^T to ensure correctness
                let m_transposed = right_operand.t();
                Some(grad_output.matmul(&m_transposed))
            } else {
                None
            };
            let grad_right = if requires_grad.1 {
                // For grad_right, we need outer product: left_operand[i] * grad_output[j] = grad_M[i,j]
                Some(compute_outer_product(left_operand, grad_output))
            } else {
                None
            };
            vec![grad_left, grad_right]
        }

        // 2D @ 1D (matrix @ vector): M @ v = result_vector
        // grad_M = grad_output @ v^T (outer product), grad_v = M^T @ grad_output
        (2, 1) => {
            let grad_left = if requires_grad.0 {
                // For grad_left, we need outer product: grad_output[i] * right_operand[j] = grad_M[i,j]
                Some(compute_outer_product(grad_output, right_operand))
            } else {
                None
            };
            let grad_right = if requires_grad.1 {
                Some(left_operand.t().matmul(grad_output))
            } else {
                None
            };
            vec![grad_left, grad_right]
        }

        // 2D @ 2D (matrix @ matrix): standard case
        (2, 2) => {
            let grad_left = if requires_grad.0 {
                Some(grad_output.matmul(&right_operand.t()))
            } else {
                None
            };
            let grad_right = if requires_grad.1 {
                Some(left_operand.t().matmul(grad_output))
            } else {
                None
            };
            vec![grad_left, grad_right]
        }

        // Special case: 3D @ 2D (mixed dimensionality) with singleton leading dimension
        (3, 2) if left_shape.dims[0] == 1 => {
            // For [1, m, k] @ [k, n] -> [m, n] (after squeezing)
            // Left gradient: [m, n] @ [n, k] -> [m, k], then unsqueeze to [1, m, k]
            // Right gradient: [1, k, m] @ [m, n] -> [k, n] (with proper broadcasting)

            let grad_left = if requires_grad.0 {
                let grad_left_2d = grad_output.matmul(&right_operand.t()); // [m, n] @ [n, k] -> [m, k]
                Some(grad_left_2d.unsqueeze(0)) // [m, k] -> [1, m, k]
            } else {
                None
            };

            let grad_right = if requires_grad.1 {
                let left_t = left_operand.transpose(1, 2); // [1, m, k] -> [1, k, m]
                let left_2d = left_t.squeeze(Some(0)); // [1, k, m] -> [k, m]
                Some(left_2d.matmul(grad_output)) // [k, m] @ [m, n] -> [k, n]
            } else {
                None
            };

            vec![grad_left, grad_right]
        }

        // Special case: 3D @ 3D with leading singleton dimension
        (3, 3) if left_shape.dims[0] == 1 || right_shape.dims[0] == 1 => {
            // Handle cases like [1, 3, 4] @ [2, 4, 5] or [3, 4, 5] @ [1, 5, 6]
            let grad_left = if requires_grad.0 {
                let right_t = transpose_last_two_dims(right_operand);
                // Ensure transpose is contiguous for correct computation
                let right_t_contiguous = if right_t.is_contiguous() {
                    right_t
                } else {
                    right_t.contiguous()
                };
                let grad_left_raw = grad_output.matmul(&right_t_contiguous);
                let reduced_grad_left =
                    reduce_gradient_to_original_shape(&grad_left_raw, &left_shape.dims);
                Some(reduced_grad_left)
            } else {
                None
            };
            let grad_right = if requires_grad.1 {
                let left_t = transpose_last_two_dims(left_operand);
                // Ensure transpose is contiguous for correct computation
                let left_t_contiguous = if left_t.is_contiguous() {
                    left_t
                } else {
                    left_t.contiguous()
                };
                let grad_right_raw = left_t_contiguous.matmul(grad_output);
                let reduced_grad_right =
                    reduce_gradient_to_original_shape(&grad_right_raw, &right_shape.dims);
                Some(reduced_grad_right)
            } else {
                None
            };
            vec![grad_left, grad_right]
        }

        // Special case: 2D @ 3D (mixed dimensionality) with singleton leading dimension
        (2, 3) if right_shape.dims[0] == 1 => {
            // For [m, k] @ [1, k, n] -> [m, n] (after squeezing)
            // Left gradient: [m, n] @ [1, n, k] -> [m, k] (with proper broadcasting)
            // Right gradient: [k, m] @ [m, n] -> [k, n], then unsqueeze to [1, k, n]

            let grad_left = if requires_grad.0 {
                // grad_output: [m, n], right_operand: [1, k, n] -> transpose to [1, n, k]
                let right_t = right_operand.transpose(1, 2); // [1, k, n] -> [1, n, k]

                // We need to handle the broadcasting manually for correct gradient computation
                // right_t is [1, n, k], we want to extract [n, k] and multiply each row of grad_output with it
                let right_2d = right_t.squeeze(Some(0)); // [1, n, k] -> [n, k]
                Some(grad_output.matmul(&right_2d)) // [m, n] @ [n, k] -> [m, k]
            } else {
                None
            };

            let grad_right = if requires_grad.1 {
                // left_operand: [m, k], grad_output: [m, n]
                // We need: [k, m] @ [m, n] -> [k, n], then unsqueeze to [1, k, n]
                let left_t = left_operand.t(); // [m, k] -> [k, m]
                let grad_right_2d = left_t.matmul(grad_output); // [k, m] @ [m, n] -> [k, n]
                                                                // Unsqueeze to match original shape [1, k, n]
                Some(grad_right_2d.unsqueeze(0))
            } else {
                None
            };

            vec![grad_left, grad_right]
        }

        // Higher dimensional cases (batched matmul)
        _ => {
            // For proper batched gradient computation, we need to use the transpose of the last two dimensions
            // but preserve the batch structure correctly
            let grad_left = if requires_grad.0 {
                // For grad_left: grad_output @ right^T
                // We transpose only the last two dimensions of right_operand
                let right_t = transpose_last_two_dims(right_operand);
                // Ensure transpose result is contiguous for correct matmul computation
                let right_t_contiguous = if right_t.is_contiguous() {
                    right_t
                } else {
                    right_t.contiguous()
                };
                let grad_left_raw = grad_output.matmul(&right_t_contiguous);

                // Apply shape reduction if broadcasting occurred
                let reduced_grad_left =
                    reduce_gradient_to_original_shape(&grad_left_raw, &left_shape.dims);
                Some(reduced_grad_left)
            } else {
                None
            };

            let grad_right = if requires_grad.1 {
                // For grad_right: left^T @ grad_output
                // We transpose only the last two dimensions of left_operand
                let left_t = transpose_last_two_dims(left_operand);
                // Ensure transpose result is contiguous for correct matmul computation
                let left_t_contiguous = if left_t.is_contiguous() {
                    left_t
                } else {
                    left_t.contiguous()
                };
                let grad_right_raw = left_t_contiguous.matmul(grad_output);

                // Apply shape reduction if broadcasting occurred
                let reduced_grad_right =
                    reduce_gradient_to_original_shape(&grad_right_raw, &right_shape.dims);
                Some(reduced_grad_right)
            } else {
                None
            };

            vec![grad_left, grad_right]
        }
    }
}

/// Compute outer product of two 1D tensors: a[i] * b[j] = result[i,j]
fn compute_outer_product(a: &Tensor, b: &Tensor) -> Tensor {
    let a_shape = a.shape();
    let b_shape = b.shape();

    assert_eq!(
        a_shape.rank(),
        1,
        "First tensor must be 1D for outer product"
    );
    assert_eq!(
        b_shape.rank(),
        1,
        "Second tensor must be 1D for outer product"
    );

    let m = a_shape.dims[0];
    let n = b_shape.dims[0];

    let mut result = Tensor::new(vec![m, n]);

    unsafe {
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        for i in 0..m {
            for j in 0..n {
                let idx = i * n + j;
                *result_ptr.add(idx) = *a_ptr.add(i) * *b_ptr.add(j);
            }
        }
    }

    result
}

/// Transpose the last two dimensions of a tensor (for batched operations)
fn transpose_last_two_dims(tensor: &Tensor) -> Tensor {
    let shape = tensor.shape();
    let rank = shape.rank();

    if rank < 2 {
        panic!(
            "Cannot transpose last two dimensions of tensor with rank {}",
            rank
        );
    }

    tensor.transpose(rank - 2, rank - 1)
}

/// Reduce gradient tensor back to the original shape by summing over broadcasted dimensions
fn reduce_gradient_to_original_shape(grad_tensor: &Tensor, original_shape: &[usize]) -> Tensor {
    let grad_shape = grad_tensor.shape();
    let grad_dims = &grad_shape.dims;

    // eprintln!("=== DEBUG reduce_gradient_to_original_shape ===");
    // eprintln!("grad_tensor shape: {:?}", grad_dims);
    // eprintln!("original_shape: {:?}", original_shape);
    // eprintln!("grad_tensor data[0..4]: {:?}", &grad_tensor.data()[0..4.min(grad_tensor.data().len())]);

    // If shapes are already the same, no reduction needed
    if grad_dims == original_shape {
        return grad_tensor.clone();
    }

    let mut result = grad_tensor.clone();
    let grad_rank = grad_dims.len();
    let orig_rank = original_shape.len();

    // For the special case where we need to reduce from 3D to 2D (e.g., [3, 4] @ [2, 4, 5])
    // The gradient tensor might be [2, 3, 4] and we need to sum to get [3, 4]
    if grad_rank == 3 && orig_rank == 2 {
        // Sum over the first dimension (batch dimension)
        result = result.sum_dims(&[0], false);
        return result;
    }

    // General case: sum over leading dimensions that were added during broadcasting
    if grad_rank > orig_rank {
        for _ in 0..(grad_rank - orig_rank) {
            result = result.sum_dims(&[0], false);
        }
    }

    // Handle dimension size mismatches (where size was 1 in original but broadcasted)
    let mut dims_to_reduce = Vec::new();
    {
        let current_shape = result.shape();
        for (i, (&current_size, &orig_size)) in current_shape
            .dims
            .iter()
            .zip(original_shape.iter())
            .enumerate()
        {
            if orig_size == 1 && current_size > 1 {
                dims_to_reduce.push(i);
            }
        }
    }

    // Apply reductions (in reverse order to maintain dimension indices)
    for &dim in dims_to_reduce.iter().rev() {
        result = result.sum_dims(&[dim], true);
    }

    result
}
