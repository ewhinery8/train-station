use crate::{tensor::ops::matmul::dispatch::MatMulKernels, Tensor};

impl Tensor {
    /// Validate matmul shapes and compute result shape
    #[inline]
    pub(super) fn validate_and_compute_matmul_shape(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Vec<usize> {
        match (left_shape.len(), right_shape.len()) {
            (1, 1) => {
                // 1D @ 1D: Dot product
                assert_eq!(
                    left_shape[0], right_shape[0],
                    "Incompatible shapes for dot product: [{:?}] @ [{:?}]",
                    left_shape, right_shape
                );
                vec![] // Scalar result
            }
            (1, 2) => {
                // 1D @ 2D: Vector-matrix multiplication
                assert_eq!(
                    left_shape[0], right_shape[0],
                    "Incompatible shapes for vector-matrix multiplication: [{:?}] @ [{:?}]",
                    left_shape, right_shape
                );
                vec![right_shape[1]] // Result is 1D
            }
            (2, 1) => {
                // 2D @ 1D: Matrix-vector multiplication
                assert_eq!(
                    left_shape[1], right_shape[0],
                    "Incompatible shapes for matrix-vector multiplication: [{:?}] @ [{:?}]",
                    left_shape, right_shape
                );
                vec![left_shape[0]] // Result is 1D
            }
            (2, 2) => {
                // 2D @ 2D: Standard matrix multiplication
                assert_eq!(
                    left_shape[1], right_shape[0],
                    "Incompatible shapes for matrix multiplication: [{:?}] @ [{:?}]",
                    left_shape, right_shape
                );
                vec![left_shape[0], right_shape[1]] // Result is 2D
            }
            _ => {
                // ND @ ND: Batched matrix multiplication
                Self::validate_and_compute_batched_matmul_shape(left_shape, right_shape)
            }
        }
    }

    /// Validate and compute shape for batched matrix multiplication
    pub(super) fn validate_and_compute_batched_matmul_shape(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> Vec<usize> {
        let debug_logging = std::env::var("TS_MATMUL_DEBUG")
            .map(|v| v == "1")
            .unwrap_or(false);

        let left_rank = left_shape.len();
        let right_rank = right_shape.len();

        // Helper to broadcast two batch shape slices aligned from the right
        fn broadcast_batches(left_batch: &[usize], right_batch: &[usize]) -> Vec<usize> {
            let max_batch_rank = left_batch.len().max(right_batch.len());
            let mut result_batch = Vec::with_capacity(max_batch_rank);
            for i in 0..max_batch_rank {
                let ldim = if i < left_batch.len() {
                    left_batch[left_batch.len() - 1 - i]
                } else {
                    1
                };
                let rdim = if i < right_batch.len() {
                    right_batch[right_batch.len() - 1 - i]
                } else {
                    1
                };
                if ldim == rdim {
                    result_batch.push(ldim);
                } else if ldim == 1 {
                    result_batch.push(rdim);
                } else if rdim == 1 {
                    result_batch.push(ldim);
                } else {
                    panic!(
                        "Incompatible batch dimensions for batched matmul: left_batch={:?}, right_batch={:?}",
                        left_batch, right_batch
                    );
                }
            }
            result_batch.reverse();
            result_batch
        }

        // Extract the contraction dimension K from left (always the last dim)
        assert!(
            left_rank >= 1 && right_rank >= 1,
            "matmul requires at least 1D tensors"
        );

        // Determine how many matrix dims each operand contributes (1 or 2),
        // preferring the standard (2,2) case when possible.
        // Choose the first matching configuration in this priority order:
        //  (2,2): [..., M, K] @ [..., K, N]
        //  (2,1): [..., M, K] @ [..., K]
        //  (1,2): [..., K]     @ [..., K, N]
        //  (1,1): [..., K]     @ [..., K]
        let (l_mat_dims, r_mat_dims) = if left_rank >= 2
            && right_rank >= 2
            && left_shape[left_rank - 1] == right_shape[right_rank - 2]
        {
            (2usize, 2usize)
        } else if left_rank >= 2
            && right_rank >= 1
            && left_shape[left_rank - 1] == right_shape[right_rank - 1]
        {
            (2usize, 1usize)
        } else if left_rank >= 1
            && right_rank >= 2
            && left_shape[left_rank - 1] == right_shape[right_rank - 2]
        {
            (1usize, 2usize)
        } else if left_rank >= 1
            && right_rank >= 1
            && left_shape[left_rank - 1] == right_shape[right_rank - 1]
        {
            (1usize, 1usize)
        } else {
            panic!(
                "Incompatible matrix dimensions for matmul: left={:?} right={:?}",
                left_shape, right_shape
            );
        };

        // Derive batch slices and output M/N based on selected configuration
        let left_batch = &left_shape[..left_rank - l_mat_dims];
        let right_batch = &right_shape[..right_rank - r_mat_dims];
        let m = if l_mat_dims == 2 {
            left_shape[left_rank - 2]
        } else {
            1
        };
        let n = if r_mat_dims == 2 {
            right_shape[right_rank - 1]
        } else {
            1
        };

        if debug_logging {
            println!(
                "[shape] config=({},{}) left={:?} right={:?} -> batches L={:?} R={:?} M={} N={}",
                l_mat_dims, r_mat_dims, left_shape, right_shape, left_batch, right_batch, m, n
            );
        }

        let mut result_shape = broadcast_batches(left_batch, right_batch);

        // Build the final result shape depending on whether any side is vector-like
        match (l_mat_dims, r_mat_dims) {
            (2, 2) => {
                result_shape.push(m);
                result_shape.push(n);
            }
            (2, 1) => {
                // matrix @ vector -> [..., M]
                result_shape.push(m);
            }
            (1, 2) => {
                // vector @ matrix -> [..., N]
                result_shape.push(n);
            }
            (1, 1) => {
                // batched dot -> [...]; if no batch dims, scalar [] handled by caller
            }
            _ => unreachable!(),
        }

        if debug_logging {
            println!(
                "[shape] final result shape = {:?} (left={:?}, right={:?})",
                result_shape, left_shape, right_shape
            );
        }
        result_shape
    }

    /// Classify how many matrix dims each operand contributes for ND matmul.
    /// Returns (l_mat_dims, r_mat_dims) where each is 1 (vector-like K) or 2 (matrix MxK / KxN).
    /// This uses the exact same precedence and checks as forward shape logic to ensure
    /// gradients follow identical broadcasting semantics.
    pub(crate) fn classify_nd_matmul_dims(
        left_shape: &[usize],
        right_shape: &[usize],
    ) -> (usize, usize) {
        let left_rank = left_shape.len();
        let right_rank = right_shape.len();

        assert!(
            left_rank >= 1 && right_rank >= 1,
            "matmul requires at least 1D tensors"
        );
        let left_k = left_shape[left_rank - 1];

        if left_rank >= 2 && right_rank >= 2 && left_k == right_shape[right_rank - 2] {
            (2, 2)
        } else if left_rank >= 2 && right_rank >= 1 && left_k == right_shape[right_rank - 1] {
            (2, 1)
        } else if left_rank >= 1 && right_rank >= 2 && left_k == right_shape[right_rank - 2] {
            (1, 2)
        } else if left_rank >= 1 && right_rank >= 1 && left_k == right_shape[right_rank - 1] {
            (1, 1)
        } else {
            panic!(
                "Incompatible matrix dimensions for matmul: left={:?} right={:?}",
                left_shape, right_shape
            );
        }
    }

    /// Dispatch batched matrix multiplication with proper stride handling for non-contiguous tensors
    pub(super) fn dispatch_batched_matmul_with_ptrs_strided(
        left_ptr: *const f32,
        right_ptr: *const f32,
        left: &Tensor,
        right: &Tensor,
        result: &mut Tensor,
        kernels: &MatMulKernels,
    ) {
        let left_shape = left.shape().dims().to_vec();
        let right_shape = right.shape().dims().to_vec();
        let result_shape = result.shape().dims().to_vec();

        let left_rank = left_shape.len();
        let right_rank = right_shape.len();
        let result_rank = result_shape.len();

        let debug_logging = std::env::var("TS_MATMUL_DEBUG")
            .map(|v| v == "1")
            .unwrap_or(false);

        // Classify how many matrix dims each side contributes (1 or 2),
        // preferring the standard (2,2) case when applicable.
        let left_k_dim = left_shape[left_rank - 1];
        let right_last_dim = right_shape[right_rank - 1];
        let right_k_dim_for_mat = if right_rank >= 2 {
            right_shape[right_rank - 2]
        } else {
            0
        };

        let (l_mat_dims, r_mat_dims) =
            if left_rank >= 2 && right_rank >= 2 && left_k_dim == right_k_dim_for_mat {
                (2usize, 2usize)
            } else if left_rank >= 2 && left_k_dim == right_last_dim {
                (2usize, 1usize)
            } else if right_rank >= 2 && left_k_dim == right_k_dim_for_mat {
                (1usize, 2usize)
            } else {
                (1usize, 1usize)
            };

        // Handle mixed cases explicitly: [K] @ [B..., K, N] (only when classified as (1,2))
        if left_rank == 1 && right_rank >= 2 && l_mat_dims == 1 && r_mat_dims == 2 {
            unsafe {
                let result_ptr = result.as_mut_ptr();

                // Vector length and output width
                let k = left_shape[0];
                let n = right_shape[right_rank - 1];

                // Strides
                let left_stride = left.strides()[0];
                let right_strides = right.strides();
                let right_row_stride = right_strides[right_rank - 2];
                let right_col_stride = right_strides[right_rank - 1];
                let result_strides = result.strides();
                let c_stride = result_strides[result_rank - 1];

                // Number of batch elements equals product of all result dims except last (N)
                let batch_dims = result_rank.saturating_sub(1);
                let batch_size: usize = if batch_dims == 0 {
                    1
                } else {
                    result_shape[..batch_dims].iter().product()
                };

                for batch_idx in 0..batch_size {
                    // Compute multi-dimensional indices over batch dims (row-major)
                    let mut batch_indices = vec![0usize; batch_dims];
                    let mut tmp = batch_idx;
                    for dim in (0..batch_dims).rev() {
                        let size = result_shape[dim];
                        if size > 0 {
                            batch_indices[dim] = tmp % size;
                            tmp /= size;
                        }
                    }

                    // Map batch indices to right tensor batch dims (align from right)
                    let mut right_offset = 0usize;
                    if right_rank > 2 {
                        let right_batch_dims = right_rank - 2;
                        let result_batch_dims = batch_dims;
                        for j in 0..right_batch_dims {
                            let rb = result_batch_dims - right_batch_dims + j;
                            let idx = if right_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % right_shape[j]
                            };
                            right_offset += idx * right_strides[j];
                        }
                    }

                    // Result offset across batch dims only
                    let mut result_offset = 0usize;
                    for d in 0..batch_dims {
                        result_offset += batch_indices[d] * result_strides[d];
                    }

                    if debug_logging && batch_idx < 2 {
                        println!(
                            "[matmul 1D@ND] batch_idx={}, right_offset={}, result_offset={}, batch_indices={:?}",
                            batch_idx, right_offset, result_offset, batch_indices
                        );
                    }

                    let left_batch_ptr = left_ptr; // broadcasted across batches
                    let right_batch_ptr = right_ptr.add(right_offset);
                    let result_batch_ptr = result_ptr.add(result_offset);

                    kernels.dispatch_vec_mat_strided(
                        left_batch_ptr,
                        right_batch_ptr,
                        result_batch_ptr,
                        k,
                        n,
                        left_stride,
                        right_row_stride,
                        right_col_stride,
                        c_stride,
                    );
                }
            }
            return;
        }

        // Handle mixed cases explicitly: [B..., M, K] @ [K] (only when classified as (2,1))
        if left_rank >= 2 && right_rank == 1 && l_mat_dims == 2 && r_mat_dims == 1 {
            unsafe {
                let result_ptr = result.as_mut_ptr();

                let m = left_shape[left_rank - 2];
                let k = left_shape[left_rank - 1];

                let left_strides = left.strides();
                let a_row_stride = left_strides[left_rank - 2];
                let a_col_stride = left_strides[left_rank - 1];
                let b_stride = right.strides()[0];
                let result_strides = result.strides();
                let c_stride = result_strides[result_rank - 1];

                let batch_dims = result_rank.saturating_sub(1);
                let batch_size: usize = if batch_dims == 0 {
                    1
                } else {
                    result_shape[..batch_dims].iter().product()
                };

                for batch_idx in 0..batch_size {
                    let mut batch_indices = vec![0usize; batch_dims];
                    let mut tmp = batch_idx;
                    for dim in (0..batch_dims).rev() {
                        let size = result_shape[dim];
                        if size > 0 {
                            batch_indices[dim] = tmp % size;
                            tmp /= size;
                        }
                    }

                    // Map batch indices to left tensor batch dims (align from right)
                    let mut left_offset = 0usize;
                    if left_rank > 2 {
                        let left_batch_dims = left_rank - 2;
                        let result_batch_dims = batch_dims;
                        for j in 0..left_batch_dims {
                            let rb = result_batch_dims - left_batch_dims + j;
                            let idx = if left_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % left_shape[j]
                            };
                            left_offset += idx * left_strides[j];
                        }
                    }

                    // Right is 1D, broadcast across batches
                    let right_offset = 0usize;

                    // Result offset across batch dims only
                    let mut result_offset = 0usize;
                    for d in 0..batch_dims {
                        result_offset += batch_indices[d] * result_strides[d];
                    }

                    if debug_logging && batch_idx < 2 {
                        println!(
                            "[matmul ND@1D] batch_idx={}, left_offset={}, result_offset={}, batch_indices={:?}",
                            batch_idx, left_offset, result_offset, batch_indices
                        );
                    }

                    let left_batch_ptr = left_ptr.add(left_offset);
                    let right_batch_ptr = right_ptr.add(right_offset);
                    let result_batch_ptr = result_ptr.add(result_offset);

                    kernels.dispatch_mat_vec_strided(
                        left_batch_ptr,
                        right_batch_ptr,
                        result_batch_ptr,
                        m,
                        k,
                        a_row_stride,
                        a_col_stride,
                        b_stride,
                        c_stride,
                    );
                }
            }
            return;
        }

        // Handle [B..., M, K] @ [B..., K] -> [B..., M] when classified as (2,1) and right has batch dims
        if l_mat_dims == 2 && r_mat_dims == 1 && right_rank > 1 {
            unsafe {
                let result_ptr = result.as_mut_ptr();

                // Dimensions
                let m = left_shape[left_rank - 2];
                let k_dim = left_shape[left_rank - 1];

                // Strides
                let left_strides = left.strides();
                let a_row_stride = left_strides[left_rank - 2];
                let a_col_stride = left_strides[left_rank - 1];
                let right_strides = right.strides();
                let b_stride = right_strides[right_rank - 1];
                let result_strides = result.strides();
                let c_stride = result_strides[result_rank - 1];

                // Batch dims are all result dims except the last (M)
                let batch_dims = result_rank.saturating_sub(1);
                let batch_size: usize = if batch_dims == 0 {
                    1
                } else {
                    result_shape[..batch_dims].iter().product()
                };

                for batch_idx in 0..batch_size {
                    // Compute multi-dimensional batch indices (row-major)
                    let mut batch_indices = vec![0usize; batch_dims];
                    let mut tmp = batch_idx;
                    for dim in (0..batch_dims).rev() {
                        let size = result_shape[dim];
                        if size > 0 {
                            batch_indices[dim] = tmp % size;
                            tmp /= size;
                        }
                    }

                    // Map to left batch dims (align from right)
                    let mut left_offset = 0usize;
                    if left_rank > 2 {
                        let left_batch_dims = left_rank - 2;
                        let result_batch_dims = batch_dims;
                        for j in 0..left_batch_dims {
                            let rb = result_batch_dims.saturating_sub(left_batch_dims) + j;
                            let idx = if left_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % left_shape[j]
                            };
                            left_offset += idx * left_strides[j];
                        }
                    }

                    // Map to right batch dims (align from right, note right vector has no trailing N)
                    let mut right_offset = 0usize;
                    if right_rank > 1 {
                        let right_batch_dims = right_rank - 1;
                        let result_batch_dims = batch_dims;
                        for j in 0..right_batch_dims {
                            let rb = result_batch_dims.saturating_sub(right_batch_dims) + j;
                            let idx = if right_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % right_shape[j]
                            };
                            right_offset += idx * right_strides[j];
                        }
                    }

                    // Result offset across batch dims only
                    let mut result_offset = 0usize;
                    for d in 0..batch_dims {
                        result_offset += batch_indices[d] * result_strides[d];
                    }

                    if debug_logging && batch_idx < 2 {
                        println!(
                            "[matmul ND@vec] batch_idx={}, left_offset={}, right_offset={}, result_offset={}, batch_indices={:?}",
                            batch_idx, left_offset, right_offset, result_offset, batch_indices
                        );
                    }

                    let left_batch_ptr = left_ptr.add(left_offset);
                    let right_batch_ptr = right_ptr.add(right_offset);
                    let result_batch_ptr = result_ptr.add(result_offset);

                    kernels.dispatch_mat_vec_strided(
                        left_batch_ptr,
                        right_batch_ptr,
                        result_batch_ptr,
                        m,
                        k_dim,
                        a_row_stride,
                        a_col_stride,
                        b_stride,
                        c_stride,
                    );
                }
            }
            return;
        }

        // Handle [B..., K] @ [B..., K, N] -> [B..., N] when classified as (1,2) and left has batch dims
        if l_mat_dims == 1 && r_mat_dims == 2 && left_rank > 1 {
            unsafe {
                let result_ptr = result.as_mut_ptr();

                // Dimensions
                let k_dim = left_shape[left_rank - 1];
                let n = right_shape[right_rank - 1];

                // Strides
                let left_strides = left.strides();
                let a_stride = left_strides[left_rank - 1];
                let right_strides = right.strides();
                let b_row_stride = right_strides[right_rank - 2];
                let b_col_stride = right_strides[right_rank - 1];
                let result_strides = result.strides();
                let c_stride = result_strides[result_rank - 1];

                // Batch dims are all result dims except the last (N)
                let batch_dims = result_rank.saturating_sub(1);
                let batch_size: usize = if batch_dims == 0 {
                    1
                } else {
                    result_shape[..batch_dims].iter().product()
                };

                for batch_idx in 0..batch_size {
                    // Compute multi-dimensional batch indices (row-major)
                    let mut batch_indices = vec![0usize; batch_dims];
                    let mut tmp = batch_idx;
                    for dim in (0..batch_dims).rev() {
                        let size = result_shape[dim];
                        if size > 0 {
                            batch_indices[dim] = tmp % size;
                            tmp /= size;
                        }
                    }

                    // Map to left batch dims (align from right, left vector has no M)
                    let mut left_offset = 0usize;
                    if left_rank > 1 {
                        let left_batch_dims = left_rank - 1;
                        let result_batch_dims = batch_dims;
                        for j in 0..left_batch_dims {
                            let rb = result_batch_dims.saturating_sub(left_batch_dims) + j;
                            let idx = if left_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % left_shape[j]
                            };
                            left_offset += idx * left_strides[j];
                        }
                    }

                    // Map to right batch dims (align from right, matrix has two trailing dims K,N)
                    let mut right_offset = 0usize;
                    if right_rank > 2 {
                        let right_batch_dims = right_rank - 2;
                        let result_batch_dims = batch_dims;
                        for j in 0..right_batch_dims {
                            let rb = result_batch_dims.saturating_sub(right_batch_dims) + j;
                            let idx = if right_shape[j] == 1 {
                                0
                            } else {
                                batch_indices[rb] % right_shape[j]
                            };
                            right_offset += idx * right_strides[j];
                        }
                    }

                    // Result offset across batch dims only
                    let mut result_offset = 0usize;
                    for d in 0..batch_dims {
                        result_offset += batch_indices[d] * result_strides[d];
                    }

                    if debug_logging && batch_idx < 2 {
                        println!(
                            "[matmul vec@ND] batch_idx={}, left_offset={}, right_offset={}, result_offset={}, batch_indices={:?}",
                            batch_idx, left_offset, right_offset, result_offset, batch_indices
                        );
                    }

                    let left_batch_ptr = left_ptr.add(left_offset);
                    let right_batch_ptr = right_ptr.add(right_offset);
                    let result_batch_ptr = result_ptr.add(result_offset);

                    kernels.dispatch_vec_mat_strided(
                        left_batch_ptr,
                        right_batch_ptr,
                        result_batch_ptr,
                        k_dim,
                        n,
                        a_stride,
                        b_row_stride,
                        b_col_stride,
                        c_stride,
                    );
                }
            }
            return;
        }

        // Matrix dimensions from the last two dimensions (ND@ND)
        let m = if left_rank >= 2 {
            left_shape[left_rank - 2]
        } else {
            1
        };
        let k = left_shape[left_rank - 1];
        let n = right_shape[right_rank - 1];

        // Batch dimensions
        let batch_size: usize = result_shape[..result_rank - 2].iter().product();

        unsafe {
            let result_ptr = result.as_mut_ptr();

            // Get strides for each tensor
            let left_strides = left.strides();
            let right_strides = right.strides();
            let result_strides = result.strides();

            // Compute matrix strides (strides for the last two dimensions)
            let left_matrix_row_stride = if left_rank >= 2 {
                left_strides[left_rank - 2]
            } else {
                0
            };
            let left_matrix_col_stride = left_strides[left_rank - 1];
            let right_matrix_row_stride = if right_rank >= 2 {
                right_strides[right_rank - 2]
            } else {
                0
            };
            let right_matrix_col_stride = right_strides[right_rank - 1];
            let result_matrix_row_stride = if result_rank >= 2 {
                result_strides[result_rank - 2]
            } else {
                0
            };
            let result_matrix_col_stride = result_strides[result_rank - 1];

            // Handle broadcasting in batch dimensions (unused but kept for clarity)
            let _left_batch_size: usize = if left_rank > 2 {
                left_shape[..left_rank - 2].iter().product()
            } else {
                1
            };
            let _right_batch_size: usize = if right_rank > 2 {
                right_shape[..right_rank - 2].iter().product()
            } else {
                1
            };

            // Process each batch using proper multi-dimensional indexing
            for batch_idx in 0..batch_size {
                // Convert linear batch index to multi-dimensional indices (row-major)
                let mut batch_indices = vec![0usize; result_rank - 2];
                let mut tmp = batch_idx;
                for dim in (0..(result_rank - 2)).rev() {
                    let size = result_shape[dim];
                    if size > 0 {
                        batch_indices[dim] = tmp % size;
                        tmp /= size;
                    }
                }

                // Calculate linear offsets for each tensor using their strides, aligning from the right
                let mut left_offset = 0usize;
                let mut right_offset = 0usize;
                let mut result_offset = 0usize;

                // Left tensor offset (handle broadcasting; align batch dims from the right)
                if left_rank > 2 {
                    let left_batch_dims = left_rank - 2; // number of batch dims in left
                    let result_batch_dims = result_rank - 2; // number of batch dims in result
                                                             // Map each left batch dim j to result batch dim index: rb = result_batch_dims - left_batch_dims + j
                    for j in 0..left_batch_dims {
                        let rb = result_batch_dims - left_batch_dims + j;
                        let idx = if left_shape[j] == 1 {
                            0
                        } else {
                            batch_indices[rb] % left_shape[j]
                        };
                        left_offset += idx * left_strides[j];
                    }
                }

                // Right tensor offset (handle broadcasting; align batch dims from the right)
                if right_rank > 2 {
                    let right_batch_dims = right_rank - 2;
                    let result_batch_dims = result_rank - 2;
                    for j in 0..right_batch_dims {
                        let rb = result_batch_dims - right_batch_dims + j;
                        let idx = if right_shape[j] == 1 {
                            0
                        } else {
                            batch_indices[rb] % right_shape[j]
                        };
                        right_offset += idx * right_strides[j];
                    }
                }

                // Result tensor offset (batch dims only)
                for d in 0..(result_rank - 2) {
                    result_offset += batch_indices[d] * result_strides[d];
                }

                if debug_logging && batch_idx < 2 {
                    println!(
                        "[matmul] batch_idx={}, left_offset={}, right_offset={}, result_offset={}, batch_indices={:?}",
                        batch_idx, left_offset, right_offset, result_offset, batch_indices
                    );
                }

                // Calculate pointers with proper stride handling
                let left_batch_ptr = left_ptr.add(left_offset);
                let right_batch_ptr = right_ptr.add(right_offset);
                let result_batch_ptr = result_ptr.add(result_offset);

                // Dispatch to strided 2D matrix multiplication kernel
                kernels.dispatch_mat_mat_strided(
                    left_batch_ptr,
                    right_batch_ptr,
                    result_batch_ptr,
                    m,
                    k,
                    n,
                    left_matrix_row_stride,
                    left_matrix_col_stride,
                    right_matrix_row_stride,
                    right_matrix_col_stride,
                    result_matrix_row_stride,
                    result_matrix_col_stride,
                );
            }
        }
    }
}
