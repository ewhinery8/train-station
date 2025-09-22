use crate::tensor::core::Tensor;

/// Reduce gradient from broadcasted shape back to original tensor shape
///
/// This function handles the reverse of broadcasting during gradient propagation.
/// When tensors are broadcasted during forward pass, their gradients need to be
/// reduced back to the original shapes during backward pass.
pub(crate) fn reduce_gradient_to_shape(grad_output: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad_output.shape().dims();

    // If shapes are already the same, no reduction needed
    if grad_shape == target_shape {
        return grad_output.clone();
    }

    let debug_logging = std::env::var("TS_MATMUL_DEBUG")
        .map(|v| v == "1")
        .unwrap_or(false);

    if debug_logging {
        println!(
            "[reduce] grad_shape={:?}, target_shape={:?}",
            grad_shape, target_shape
        );
    }

    // Early-case: if sizes match and we can insert 1-dims to match target, just reshape
    let current_size: usize = grad_shape.iter().product();
    let target_size: usize = target_shape.iter().product();
    if current_size == target_size && can_insert_ones_to_match(grad_shape, target_shape) {
        if debug_logging {
            println!(
                "[reduce] early reshape via inserting ones: grad_shape={:?} -> target_shape={:?}",
                grad_shape, target_shape
            );
        }
        return grad_output.reshape(target_shape.iter().map(|&d| d as i32).collect());
    }

    // Handle empty target shape (scalar)
    if target_shape.is_empty() {
        let mut result = grad_output.clone();
        while !result.shape().dims().is_empty() {
            result = result.sum_dims(&[0], false);
        }
        return result;
    }

    let mut result = grad_output.clone();
    let _grad_rank = grad_shape.len();
    let target_rank = target_shape.len();

    // Step 1: Sum over leading dimensions if grad has more dimensions than target
    while result.shape().dims().len() > target_rank {
        if debug_logging {
            println!(
                "[reduce] summing leading dim, current shape={:?}",
                result.shape().dims()
            );
        }
        result = result.sum_dims(&[0], false);
    }

    // Step 2: Handle case where target has more dimensions (pad with 1s conceptually)
    let current_dims = result.shape().dims().to_vec();
    let current_rank = current_dims.len();

    if current_rank < target_rank {
        // Target has more dimensions - this shouldn't happen in normal broadcasting
        // but handle it by reshaping with leading 1s
        let mut new_shape = vec![1; target_rank - current_rank];
        new_shape.extend_from_slice(&current_dims);
        result = result.reshape(new_shape.iter().map(|&d| d as i32).collect());
    }

    // Step 3: Handle dimension reduction properly for broadcasting cases
    let current_dims = result.shape().dims().to_vec();

    if debug_logging {
        println!(
            "[reduce] current_dims={:?}, target_shape={:?}",
            current_dims, target_shape
        );
    }

    // For broadcasting reduction, we need to be more careful about dimension alignment
    // The key insight is that broadcasting works right-to-left, so we align from the right

    let current_rank = current_dims.len();
    let target_rank = target_shape.len();

    if current_rank == target_rank {
        // Same rank - check each dimension for reduction needs
        let mut axes_to_sum = Vec::new();
        for (i, (&current_dim, &target_dim)) in
            current_dims.iter().zip(target_shape.iter()).enumerate()
        {
            if target_dim == 1 && current_dim > 1 {
                axes_to_sum.push(i);
            }
        }

        // Sum from highest to lowest to avoid index shifting
        for &axis in axes_to_sum.iter().rev() {
            if debug_logging {
                println!("[reduce] summing axis {} with keepdim=true", axis);
            }
            result = result.sum_dims(&[axis], true);
        }
    } else if current_rank < target_rank {
        // Need to add leading dimensions
        let mut new_shape = vec![1; target_rank - current_rank];
        new_shape.extend_from_slice(&current_dims);
        result = result.reshape(new_shape.iter().map(|&d| d as i32).collect());

        // Now check for reductions
        let reshaped_dims = result.shape().dims().to_vec();
        let mut axes_to_sum = Vec::new();
        for (i, (&reshaped_dim, &target_dim)) in
            reshaped_dims.iter().zip(target_shape.iter()).enumerate()
        {
            if target_dim == 1 && reshaped_dim > 1 {
                axes_to_sum.push(i);
            }
        }

        for &axis in axes_to_sum.iter().rev() {
            if debug_logging {
                println!("[reduce] summing axis {} with keepdim=true", axis);
            }
            result = result.sum_dims(&[axis], true);
        }
    } else {
        // current_rank > target_rank - need to reduce leading dimensions
        let excess_dims = current_rank - target_rank;

        // Sum over excess leading dimensions
        for _ in 0..excess_dims {
            if debug_logging {
                println!(
                    "[reduce] summing leading dim, current shape={:?}",
                    result.shape().dims()
                );
            }
            result = result.sum_dims(&[0], false);
        }

        // Now check remaining dimensions for reductions
        let reduced_dims = result.shape().dims().to_vec();
        let mut axes_to_sum = Vec::new();
        for (i, (&reduced_dim, &target_dim)) in
            reduced_dims.iter().zip(target_shape.iter()).enumerate()
        {
            if target_dim == 1 && reduced_dim > 1 {
                axes_to_sum.push(i);
            }
        }

        for &axis in axes_to_sum.iter().rev() {
            if debug_logging {
                println!("[reduce] summing axis {} with keepdim=true", axis);
            }
            result = result.sum_dims(&[axis], true);
        }
    }

    // Step 4: Final reshape to exact target shape
    let current_size: usize = result.shape().dims().iter().product();
    let target_size: usize = target_shape.iter().product();

    if current_size == target_size {
        // Sizes match - we can reshape directly
        result = result.reshape(target_shape.iter().map(|&d| d as i32).collect());
    } else {
        // Special case: check if we can insert 1-dimensions to match
        // This happens in broadcasting cases like [2,5,6] -> [2,1,5,6]
        let current_dims = result.shape().dims().to_vec();

        // Try to find a way to insert 1s to make the shapes compatible
        if target_size == current_size {
            // This should have been caught above, but just in case
            result = result.reshape(target_shape.iter().map(|&d| d as i32).collect());
        } else if can_insert_ones_to_match(&current_dims, target_shape) {
            // We can insert 1-dimensions to match the target shape
            if debug_logging {
                println!("[reduce] Inserting 1-dimensions to match target shape");
            }
            result = result.reshape(target_shape.iter().map(|&d| d as i32).collect());
        } else {
            if debug_logging {
                println!(
                    "[reduce] Size mismatch: current_size={}, target_size={}, current_shape={:?}",
                    current_size,
                    target_size,
                    result.shape().dims()
                );
            }
            panic!(
                "Cannot reshape tensor of size {} to shape {:?} (size {})",
                current_size, target_shape, target_size
            );
        }
    }

    if debug_logging {
        println!("[reduce] final result_shape={:?}", result.shape().dims());
    }

    result
}

/// Check if we can insert 1-dimensions to transform current_dims into target_shape
fn can_insert_ones_to_match(current_dims: &[usize], target_shape: &[usize]) -> bool {
    // Check if target_shape can be obtained by inserting 1s into current_dims
    // This is true if all non-1 dimensions in target_shape match current_dims in order

    let mut current_idx = 0;
    for &target_dim in target_shape {
        if target_dim == 1 {
            // Skip 1-dimensions in target (these can be inserted)
            continue;
        }

        if current_idx >= current_dims.len() {
            // Ran out of current dimensions
            return false;
        }

        if current_dims[current_idx] != target_dim {
            // Dimension mismatch
            return false;
        }

        current_idx += 1;
    }

    // Check that we've consumed all current dimensions
    current_idx == current_dims.len()
}

/// Reduce a matmul gradient tensor's batch dimensions to match an operand's batch shape,
/// preserving the last `keep_last` matrix dimensions (typically 2 for K,N or M,K).
/// This aligns batch axes by value, reorders batch axes when needed, and sums over
/// axes that correspond to broadcasted/missing dimensions in the operand.
pub(crate) fn reduce_matmul_grad_to_operand_shape(
    grad: &Tensor,
    operand_shape: &[usize],
    keep_last: usize,
) -> Tensor {
    let g_dims = grad.shape().dims().to_vec();
    let g_rank = g_dims.len();
    let g_batch_len = g_rank.saturating_sub(keep_last);
    let op_rank = operand_shape.len();
    let op_batch_len = op_rank.saturating_sub(keep_last);

    // Fast path: already matches
    if g_batch_len == op_batch_len && g_dims[..g_batch_len] == operand_shape[..op_batch_len] {
        return grad.clone();
    }

    // We'll work on a copy, and only reorder/sum batch axes (prefix of dims)
    let mut result = grad.clone();

    // Helper: swap two batch axes using transpose and materialize contiguity
    // Implemented as a small inline function to avoid borrow checker conflicts
    fn swap_axes(mut t: Tensor, a: usize, b: usize, keep_last: usize) -> Tensor {
        if a == b {
            return t;
        }
        let rank_now = t.shape().dims().len();
        debug_assert!(a < rank_now - keep_last && b < rank_now - keep_last);
        t = t.transpose(a, b);
        t.contiguous()
    }

    // 1) Bring batch axes that should be kept (matching operand batch dims) into place.
    // Strategy: For each operand batch dim j (0..op_batch_len), find a batch axis in result
    // whose size equals operand_shape[j] (prefer exact matches), otherwise keep a 1-dim.
    // Move that axis to position j via swaps. Remaining batch axes will be summed away.
    // Use right-aligned matching like broadcasting semantics.
    for (j, &target) in operand_shape.iter().enumerate().take(op_batch_len) {
        let current_dims = result.shape().dims().to_vec();
        let batch_end = current_dims.len().saturating_sub(keep_last);

        // Map operand batch dim j to corresponding grad batch dim (right-aligned)
        let grad_idx = if g_batch_len >= op_batch_len {
            // Grad has more batch dims, map j to (g_batch_len - op_batch_len + j)
            g_batch_len - op_batch_len + j
        } else {
            // Grad has fewer batch dims, map j to j
            j
        };

        let mut found: Option<usize> = None;

        // Search for exact match starting from the mapped position
        if grad_idx < batch_end && current_dims[grad_idx] == target {
            found = Some(grad_idx);
        }

        // If no exact match, try to find a 1-dim we can keep at the mapped position
        if found.is_none() && grad_idx < batch_end && current_dims[grad_idx] == 1 {
            found = Some(grad_idx);
        }

        // If still not found, search for exact match in all remaining batch axes
        if found.is_none() {
            for (idx, &dim) in current_dims.iter().enumerate().take(batch_end) {
                if dim == target {
                    found = Some(idx);
                    break;
                }
            }
        }

        // If still not found, try to find a 1-dim we can keep
        if found.is_none() {
            for (idx, &dim) in current_dims.iter().enumerate().take(batch_end) {
                if dim == 1 {
                    found = Some(idx);
                    break;
                }
            }
        }

        // If still not found, just pick the next axis j (we'll sum it later)
        let src = found.unwrap_or(j);
        if src != j {
            result = swap_axes(result, src, j, keep_last);
        }
    }

    // 2) Sum away any remaining batch axes beyond op_batch_len
    while result.shape().dims().len() > op_batch_len + keep_last {
        // Always sum at position op_batch_len (first extra batch axis) to avoid re-indexing
        result = result.sum_dims(&[op_batch_len], false);
    }

    // 3) For axes where operand batch dim is 1 but current is >1, sum with keepdim=true
    let mut dims_now = result.shape().dims().to_vec();
    for j in 0..op_batch_len {
        if operand_shape[j] == 1 && dims_now[j] > 1 {
            result = result.sum_dims(&[j], true);
            dims_now = result.shape().dims().to_vec();
        }
    }

    // 4) Final reshape to exact operand shape (should be safe/size-equal now)
    reduce_gradient_to_shape(&result, operand_shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_gradient_same_shape() {
        let grad = Tensor::ones(vec![2, 3]);
        let target_shape = vec![2, 3];
        let result = reduce_gradient_to_shape(&grad, &target_shape);
        assert_eq!(result.shape().dims(), target_shape);
    }

    #[test]
    fn test_reduce_gradient_size_one_broadcast() {
        // Simulate gradient from broadcasting (2,1) -> (2,3)
        let grad = Tensor::ones(vec![2, 3]);
        let target_shape = vec![2, 1];
        let result = reduce_gradient_to_shape(&grad, &target_shape);
        assert_eq!(result.shape().dims(), target_shape);
        // Each row should sum to 3.0 (summed over broadcasted dimension)
        assert_eq!(result.get(&[0, 0]), 3.0);
        assert_eq!(result.get(&[1, 0]), 3.0);
    }

    #[test]
    fn test_reduce_gradient_rank_difference() {
        // Simulate gradient from broadcasting (3,) -> (2,3)
        let grad = Tensor::ones(vec![2, 3]);
        let target_shape = vec![3];
        let result = reduce_gradient_to_shape(&grad, &target_shape);
        assert_eq!(result.shape().dims(), target_shape);
        // Each element should sum to 2.0 (summed over added dimension)
        assert_eq!(result.get(&[0]), 2.0);
        assert_eq!(result.get(&[1]), 2.0);
        assert_eq!(result.get(&[2]), 2.0);
    }
}
