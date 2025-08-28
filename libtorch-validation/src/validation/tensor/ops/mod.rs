//! Tensor operations validation module
//!
//! Contains validation methods for all tensor mathematical operations.

pub mod add;
pub mod broadcasting_gradients;
pub mod complex_broadcasting_gradients;
pub mod div;
pub mod exp;
pub mod leaky_relu;
pub mod log;
pub mod matmul;
pub mod mul;
pub mod non_contiguous_gradients;
pub mod pow;
pub mod relu;
pub mod sigmoid;
pub mod softmax;
pub mod sqrt;
pub mod sub;
pub mod sum;
pub mod tanh;
pub mod view_gradient_accumulation;

// Additional operation validation modules for future expansion:
// pub mod elementwise; // Element-wise operations (exp, log, etc.) validation
// pub mod reduce;      // Reduction operations (sum, mean, max, min) validation
