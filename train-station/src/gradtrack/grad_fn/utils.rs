use crate::tensor::core::Tensor;

/// Reduce gradient from broadcasted shape back to original tensor shape
///
/// This function handles the reverse of broadcasting during gradient propagation.
/// When tensors are broadcasted during forward pass, their gradients need to be
/// reduced back to the original shapes during backward pass.
///
/// # Arguments
///
/// * `grad_output` - Gradient tensor from the operation (potentially broadcasted shape)
/// * `target_shape` - Original shape to reduce the gradient back to
///
/// # Returns
///
/// Gradient tensor reduced to the target shape
///
/// # Algorithm
///
/// 1. **Rank Reduction**: If target has fewer dimensions, sum over leading dimensions
/// 2. **Size-1 Reduction**: If target dimension was size 1 but broadcasted, sum and keep dimension
/// 3. **Final Reshape**: Ensure exact target shape match
pub(crate) fn reduce_gradient_to_shape(grad_output: &Tensor, target_shape: &[usize]) -> Tensor {
    let grad_shape = grad_output.shape().dims.clone();

    // If shapes are already the same, no reduction needed
    if grad_shape == target_shape {
        return grad_output.clone();
    }

    let mut result = grad_output.clone();

    // Handle case where target has fewer dimensions
    // Sum over leading dimensions that were added during broadcasting
    let rank_diff = grad_shape.len() as i32 - target_shape.len() as i32;
    if rank_diff > 0 {
        for _ in 0..rank_diff {
            result = result.sum_dims(&[0], false);
        }
    }

    // Now handle dimensions that were size 1 in original but broadcasted
    // For each dimension, if target was size 1 but current is > 1, sum it
    let current_shape = result.shape().dims.clone();
    for (i, (&current_dim, &target_dim)) in
        current_shape.iter().zip(target_shape.iter()).enumerate()
    {
        if target_dim == 1 && current_dim > 1 {
            // This dimension was broadcasted from size 1, sum it
            result = result.sum_dims(&[i], true); // keepdim=true to maintain rank
        }
    }

    // Final reshape to ensure exact target shape
    if result.shape().dims != target_shape {
        result = result.reshape(target_shape.iter().map(|&d| d as i32).collect());
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reduce_gradient_same_shape() {
        let grad = Tensor::ones(vec![2, 3]);
        let target_shape = vec![2, 3];
        let result = reduce_gradient_to_shape(&grad, &target_shape);
        assert_eq!(result.shape().dims, target_shape);
    }

    #[test]
    fn test_reduce_gradient_size_one_broadcast() {
        // Simulate gradient from broadcasting (2,1) -> (2,3)
        let grad = Tensor::ones(vec![2, 3]);
        let target_shape = vec![2, 1];
        let result = reduce_gradient_to_shape(&grad, &target_shape);
        assert_eq!(result.shape().dims, target_shape);
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
        assert_eq!(result.shape().dims, target_shape);
        // Each element should sum to 2.0 (summed over added dimension)
        assert_eq!(result.get(&[0]), 2.0);
        assert_eq!(result.get(&[1]), 2.0);
        assert_eq!(result.get(&[2]), 2.0);
    }
}
