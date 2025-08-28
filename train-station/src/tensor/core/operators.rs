//! Tensor operator overloading for natural mathematical expressions
//!
//! This module provides comprehensive operator overloading for tensor operations,
//! enabling natural mathematical expressions with tensors and scalars. All operators
//! are zero-cost abstractions that delegate to the underlying tensor operations.
//!
//! # Supported Operations
//!
//! ## Tensor-Tensor Operations
//! - **Addition**: `Tensor + Tensor`, `&Tensor + &Tensor`, `Tensor + &Tensor`, `&Tensor + Tensor`
//! - **Subtraction**: `Tensor - Tensor`, `&Tensor - &Tensor`, `Tensor - &Tensor`, `&Tensor - Tensor`
//! - **Multiplication**: `Tensor * Tensor`, `&Tensor * &Tensor`, `Tensor * &Tensor`, `&Tensor * Tensor`
//! - **Division**: `Tensor / Tensor`, `&Tensor / &Tensor`, `Tensor / &Tensor`, `&Tensor / Tensor`
//!
//! ## Tensor-Scalar Operations
//! - **Addition**: `Tensor + f32`, `&Tensor + f32`, `f32 + Tensor`, `f32 + &Tensor`
//! - **Subtraction**: `Tensor - f32`, `&Tensor - f32`, `f32 - Tensor`, `f32 - &Tensor`
//! - **Multiplication**: `Tensor * f32`, `&Tensor * f32`, `f32 * Tensor`, `f32 * &Tensor`
//! - **Division**: `Tensor / f32`, `&Tensor / f32`, `f32 / Tensor`, `f32 / &Tensor`
//!
//! ## Assignment Operations
//! - **In-place addition**: `Tensor += Tensor`, `Tensor += &Tensor`, `Tensor += f32`
//! - **In-place subtraction**: `Tensor -= Tensor`, `Tensor -= &Tensor`, `Tensor -= f32`
//! - **In-place multiplication**: `Tensor *= Tensor`, `Tensor *= &Tensor`, `Tensor *= f32`
//! - **In-place division**: `Tensor /= Tensor`, `Tensor /= &Tensor`, `Tensor /= f32`
//!
//! ## Unary Operations
//! - **Negation**: `-Tensor`, `-&Tensor`
//!
//! # Performance Characteristics
//!
//! - **Zero-Cost Abstractions**: All operators have no runtime overhead
//! - **SIMD Optimization**: Underlying operations use SIMD acceleration
//! - **Memory Efficiency**: Operations are optimized for cache performance
//! - **Thread Safety**: All operations are thread-safe
//!
//! # Examples
//!
//! ## Basic Tensor Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
//!
//! // Tensor addition
//! let result = a.clone() + b.clone();
//! assert_eq!(result.get(&[0, 0]), 6.0);
//!
//! // Element-wise multiplication
//! let result = a.clone() * b.clone();
//! assert_eq!(result.get(&[0, 0]), 5.0);
//! ```
//!
//! ## Scalar Operations
//!
//! ```
//! use train_station::Tensor;
//!
//! let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // Tensor + scalar
//! let result = tensor.clone() + 5.0;
//! assert_eq!(result.get(&[0, 0]), 6.0);
//!
//! // Scalar + tensor
//! let result = 5.0 + tensor.clone();
//! assert_eq!(result.get(&[0, 0]), 6.0);
//!
//! // Tensor * scalar
//! let result = tensor.clone() * 3.0;
//! assert_eq!(result.get(&[0, 0]), 3.0);
//! ```
//!
//! ## Compound Expressions
//!
//! ```
//! use train_station::Tensor;
//!
//! let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//! let b = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();
//!
//! // Complex mathematical expression
//! let result = (a.clone() + b.clone()) * 2.0 - 1.0;
//! assert_eq!(result.get(&[0, 0]), 5.0); // (1+2)*2-1 = 5
//! ```
//!
//! ## Assignment Operators
//!
//! ```
//! use train_station::Tensor;
//!
//! let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
//!
//! // In-place operations
//! tensor += 5.0;
//! assert_eq!(tensor.get(&[0, 0]), 6.0);
//!
//! tensor *= 2.0;
//! assert_eq!(tensor.get(&[0, 0]), 12.0);
//! ```
//!
//! # Thread Safety
//!
//! All operator implementations are thread-safe and can be used concurrently
//! across multiple threads. Operations on different tensors can be performed
//! simultaneously without synchronization.

use super::Tensor;

use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

// ===== Tensor-Tensor Operations =====

/// Tensor addition operator implementations
///
/// Provides addition operations between tensors with various reference combinations.
/// All implementations delegate to the underlying `add_tensor` method for optimal performance.
impl Add for Tensor {
    type Output = Tensor;

    /// Adds two tensors element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    fn add(self, other: Tensor) -> Tensor {
        self.add_tensor(&other)
    }
}

impl Add for &Tensor {
    type Output = Tensor;

    /// Adds two tensors element-wise (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    fn add(self, other: &Tensor) -> Tensor {
        self.add_tensor(other)
    }
}

impl Add<&Tensor> for Tensor {
    type Output = Tensor;

    /// Adds a tensor and a tensor reference element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    fn add(self, other: &Tensor) -> Tensor {
        self.add_tensor(other)
    }
}

impl Add<Tensor> for &Tensor {
    type Output = Tensor;

    /// Adds a tensor reference and a tensor element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise sum
    fn add(self, other: Tensor) -> Tensor {
        self.add_tensor(&other)
    }
}

/// Tensor addition assignment operator implementations
///
/// Provides in-place addition operations between tensors.
/// All implementations delegate to the underlying `add_tensor` method.
impl AddAssign for Tensor {
    /// Adds another tensor to this tensor in-place
    fn add_assign(&mut self, other: Tensor) {
        *self = self.add_tensor(&other);
    }
}

impl AddAssign<&Tensor> for Tensor {
    /// Adds another tensor reference to this tensor in-place
    fn add_assign(&mut self, other: &Tensor) {
        *self = self.add_tensor(other);
    }
}

/// Tensor subtraction operator implementations
///
/// Provides subtraction operations between tensors with various reference combinations.
/// All implementations delegate to the underlying `sub_tensor` method for optimal performance.
impl Sub for Tensor {
    type Output = Tensor;

    /// Subtracts two tensors element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference
    fn sub(self, other: Tensor) -> Tensor {
        self.sub_tensor(&other)
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    /// Subtracts two tensors element-wise (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference
    fn sub(self, other: &Tensor) -> Tensor {
        self.sub_tensor(other)
    }
}

impl Sub<&Tensor> for Tensor {
    type Output = Tensor;

    /// Subtracts a tensor reference from a tensor element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference
    fn sub(self, other: &Tensor) -> Tensor {
        self.sub_tensor(other)
    }
}

impl Sub<Tensor> for &Tensor {
    type Output = Tensor;

    /// Subtracts a tensor from a tensor reference element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise difference
    fn sub(self, other: Tensor) -> Tensor {
        self.sub_tensor(&other)
    }
}

/// Tensor subtraction assignment operator implementations
///
/// Provides in-place subtraction operations between tensors.
/// All implementations delegate to the underlying `sub_tensor` method.
impl SubAssign for Tensor {
    /// Subtracts another tensor from this tensor in-place
    fn sub_assign(&mut self, other: Tensor) {
        *self = self.sub_tensor(&other);
    }
}

impl SubAssign<&Tensor> for Tensor {
    /// Subtracts another tensor reference from this tensor in-place
    fn sub_assign(&mut self, other: &Tensor) {
        *self = self.sub_tensor(other);
    }
}

/// Tensor multiplication operator implementations
///
/// Provides element-wise multiplication operations between tensors with various reference combinations.
/// All implementations delegate to the underlying `mul_tensor` method for optimal performance.
impl Mul for Tensor {
    type Output = Tensor;

    /// Multiplies two tensors element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product
    fn mul(self, other: Tensor) -> Tensor {
        self.mul_tensor(&other)
    }
}

impl Mul for &Tensor {
    type Output = Tensor;

    /// Multiplies two tensors element-wise (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul_tensor(other)
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    /// Multiplies a tensor and a tensor reference element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product
    fn mul(self, other: &Tensor) -> Tensor {
        self.mul_tensor(other)
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    /// Multiplies a tensor reference and a tensor element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise product
    fn mul(self, other: Tensor) -> Tensor {
        self.mul_tensor(&other)
    }
}

/// Tensor multiplication assignment operator implementations
///
/// Provides in-place multiplication operations between tensors.
/// All implementations delegate to the underlying `mul_tensor` method.
impl MulAssign for Tensor {
    /// Multiplies this tensor by another tensor in-place
    fn mul_assign(&mut self, other: Tensor) {
        *self = self.mul_tensor(&other);
    }
}

impl MulAssign<&Tensor> for Tensor {
    /// Multiplies this tensor by another tensor reference in-place
    fn mul_assign(&mut self, other: &Tensor) {
        *self = self.mul_tensor(other);
    }
}

/// Tensor division operator implementations
///
/// Provides element-wise division operations between tensors with various reference combinations.
/// All implementations delegate to the underlying `div_tensor` method for optimal performance.
impl Div for Tensor {
    type Output = Tensor;

    /// Divides two tensors element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise quotient
    fn div(self, other: Tensor) -> Tensor {
        self.div_tensor(&other)
    }
}

impl Div for &Tensor {
    type Output = Tensor;

    /// Divides two tensors element-wise (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise quotient
    fn div(self, other: &Tensor) -> Tensor {
        self.div_tensor(other)
    }
}

impl Div<&Tensor> for Tensor {
    type Output = Tensor;

    /// Divides a tensor by a tensor reference element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise quotient
    fn div(self, other: &Tensor) -> Tensor {
        self.div_tensor(other)
    }
}

impl Div<Tensor> for &Tensor {
    type Output = Tensor;

    /// Divides a tensor reference by a tensor element-wise
    ///
    /// # Returns
    ///
    /// A new tensor containing the element-wise quotient
    fn div(self, other: Tensor) -> Tensor {
        self.div_tensor(&other)
    }
}

/// Tensor division assignment operator implementations
///
/// Provides in-place division operations between tensors.
/// All implementations delegate to the underlying `div_tensor` method.
impl DivAssign for Tensor {
    /// Divides this tensor by another tensor in-place
    fn div_assign(&mut self, other: Tensor) {
        *self = self.div_tensor(&other);
    }
}

impl DivAssign<&Tensor> for Tensor {
    /// Divides this tensor by another tensor reference in-place
    fn div_assign(&mut self, other: &Tensor) {
        *self = self.div_tensor(other);
    }
}

// ===== Scalar Operations =====

/// Tensor-scalar addition operator implementations
///
/// Provides addition operations between tensors and scalars.
/// All implementations delegate to the underlying `add_scalar` method.
impl Add<f32> for Tensor {
    type Output = Tensor;

    /// Adds a scalar to each element of the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar added to each element
    fn add(self, scalar: f32) -> Tensor {
        self.add_scalar(scalar)
    }
}

impl Add<f32> for &Tensor {
    type Output = Tensor;

    /// Adds a scalar to each element of the tensor (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar added to each element
    fn add(self, scalar: f32) -> Tensor {
        self.add_scalar(scalar)
    }
}

/// Scalar-tensor addition operator implementations
///
/// Provides addition operations between scalars and tensors.
/// All implementations delegate to the underlying `add_scalar` method.
impl Add<Tensor> for f32 {
    type Output = Tensor;

    /// Adds a scalar to each element of the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar added to each element
    fn add(self, tensor: Tensor) -> Tensor {
        tensor.add_scalar(self)
    }
}

impl Add<&Tensor> for f32 {
    type Output = Tensor;

    /// Adds a scalar to each element of the tensor (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar added to each element
    fn add(self, tensor: &Tensor) -> Tensor {
        tensor.add_scalar(self)
    }
}

/// Tensor-scalar addition assignment operator implementations
///
/// Provides in-place addition operations between tensors and scalars.
impl AddAssign<f32> for Tensor {
    /// Adds a scalar to each element of this tensor in-place
    fn add_assign(&mut self, scalar: f32) {
        *self = self.add_scalar(scalar);
    }
}

/// Tensor-scalar subtraction operator implementations
///
/// Provides subtraction operations between tensors and scalars.
/// All implementations delegate to the underlying `sub_scalar` method.
impl Sub<f32> for Tensor {
    type Output = Tensor;

    /// Subtracts a scalar from each element of the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar subtracted from each element
    fn sub(self, scalar: f32) -> Tensor {
        self.sub_scalar(scalar)
    }
}

impl Sub<f32> for &Tensor {
    type Output = Tensor;

    /// Subtracts a scalar from each element of the tensor (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar subtracted from each element
    fn sub(self, scalar: f32) -> Tensor {
        self.sub_scalar(scalar)
    }
}

/// Scalar-tensor subtraction operator implementations
///
/// Provides subtraction operations between scalars and tensors.
/// Computes `scalar - tensor` by negating the tensor and adding the scalar.
impl Sub<Tensor> for f32 {
    type Output = Tensor;

    /// Subtracts each element of the tensor from the scalar
    ///
    /// # Returns
    ///
    /// A new tensor with each element subtracted from the scalar
    fn sub(self, tensor: Tensor) -> Tensor {
        // For scalar - tensor, we need to negate the tensor and add the scalar
        // This is equivalent to: scalar + (-tensor)
        let mut result = tensor;
        result.negate_inplace();
        result.add_scalar(self)
    }
}

impl Sub<&Tensor> for f32 {
    type Output = Tensor;

    /// Subtracts each element of the tensor from the scalar (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with each element subtracted from the scalar
    fn sub(self, tensor: &Tensor) -> Tensor {
        // For scalar - tensor, we need to negate the tensor and add the scalar
        let mut result = tensor.clone();
        result.negate_inplace();
        result.add_scalar(self)
    }
}

/// Tensor-scalar subtraction assignment operator implementations
///
/// Provides in-place subtraction operations between tensors and scalars.
impl SubAssign<f32> for Tensor {
    /// Subtracts a scalar from each element of this tensor in-place
    fn sub_assign(&mut self, scalar: f32) {
        *self = self.sub_scalar(scalar);
    }
}

/// Tensor-scalar multiplication operator implementations
///
/// Provides multiplication operations between tensors and scalars.
/// All implementations delegate to the underlying `mul_scalar` method.
impl Mul<f32> for Tensor {
    type Output = Tensor;

    /// Multiplies each element of the tensor by a scalar
    ///
    /// # Returns
    ///
    /// A new tensor with each element multiplied by the scalar
    fn mul(self, scalar: f32) -> Tensor {
        self.mul_scalar(scalar)
    }
}

impl Mul<f32> for &Tensor {
    type Output = Tensor;

    /// Multiplies each element of the tensor by a scalar (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with each element multiplied by the scalar
    fn mul(self, scalar: f32) -> Tensor {
        self.mul_scalar(scalar)
    }
}

/// Scalar-tensor multiplication operator implementations
///
/// Provides multiplication operations between scalars and tensors.
/// All implementations delegate to the underlying `mul_scalar` method.
impl Mul<Tensor> for f32 {
    type Output = Tensor;

    /// Multiplies each element of the tensor by a scalar
    ///
    /// # Returns
    ///
    /// A new tensor with each element multiplied by the scalar
    fn mul(self, tensor: Tensor) -> Tensor {
        tensor.mul_scalar(self)
    }
}

impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    /// Multiplies each element of the tensor by a scalar (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with each element multiplied by the scalar
    fn mul(self, tensor: &Tensor) -> Tensor {
        tensor.mul_scalar(self)
    }
}

/// Tensor-scalar multiplication assignment operator implementations
///
/// Provides in-place multiplication operations between tensors and scalars.
impl MulAssign<f32> for Tensor {
    /// Multiplies each element of this tensor by a scalar in-place
    fn mul_assign(&mut self, scalar: f32) {
        *self = self.mul_scalar(scalar);
    }
}

/// Tensor-scalar division operator implementations
///
/// Provides division operations between tensors and scalars.
/// All implementations delegate to the underlying `div_scalar` method.
impl Div<f32> for Tensor {
    type Output = Tensor;

    /// Divides each element of the tensor by a scalar
    ///
    /// # Returns
    ///
    /// A new tensor with each element divided by the scalar
    fn div(self, scalar: f32) -> Tensor {
        self.div_scalar(scalar)
    }
}

impl Div<f32> for &Tensor {
    type Output = Tensor;

    /// Divides each element of the tensor by a scalar (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with each element divided by the scalar
    fn div(self, scalar: f32) -> Tensor {
        self.div_scalar(scalar)
    }
}

/// Scalar-tensor division operator implementations
///
/// Provides division operations between scalars and tensors.
/// Computes `scalar / tensor` by computing the reciprocal of the tensor and multiplying by the scalar.
impl Div<Tensor> for f32 {
    type Output = Tensor;

    /// Divides a scalar by each element of the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar divided by each element
    fn div(self, tensor: Tensor) -> Tensor {
        // For scalar / tensor, we need to compute scalar / each element
        // This is equivalent to: scalar * (1 / tensor)
        tensor.pow_scalar(-1.0).mul_scalar(self)
    }
}

impl Div<&Tensor> for f32 {
    type Output = Tensor;

    /// Divides a scalar by each element of the tensor (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with the scalar divided by each element
    fn div(self, tensor: &Tensor) -> Tensor {
        // For scalar / tensor, we need to compute scalar / each element
        tensor.pow_scalar(-1.0).mul_scalar(self)
    }
}

/// Tensor-scalar division assignment operator implementations
///
/// Provides in-place division operations between tensors and scalars.
impl DivAssign<f32> for Tensor {
    /// Divides each element of this tensor by a scalar in-place
    fn div_assign(&mut self, scalar: f32) {
        *self = self.div_scalar(scalar);
    }
}

// ===== Negation =====

use std::ops::Neg;

/// Tensor negation operator implementations
///
/// Provides unary negation operations for tensors.
/// All implementations delegate to the underlying `mul_scalar` method with -1.0.
impl Neg for Tensor {
    type Output = Tensor;

    /// Negates each element of the tensor
    ///
    /// # Returns
    ///
    /// A new tensor with each element negated
    fn neg(self) -> Tensor {
        self.mul_scalar(-1.0)
    }
}

impl Neg for &Tensor {
    type Output = Tensor;

    /// Negates each element of the tensor (reference version)
    ///
    /// # Returns
    ///
    /// A new tensor with each element negated
    fn neg(self) -> Tensor {
        self.mul_scalar(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== Operator Overloading Tests =====

    /// Test tensor addition operator overloading
    ///
    /// Verifies that all tensor addition operator combinations work correctly:
    /// Tensor + Tensor, &Tensor + &Tensor, Tensor + &Tensor, &Tensor + Tensor,
    /// and assignment operators (+=).
    #[test]
    fn test_tensor_addition_operators() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

        // Tensor + Tensor
        let result = a.clone() + b.clone();
        assert_eq!(result.get(&[0, 0]), 6.0);
        assert_eq!(result.get(&[0, 1]), 8.0);
        assert_eq!(result.get(&[1, 0]), 10.0);
        assert_eq!(result.get(&[1, 1]), 12.0);

        // &Tensor + &Tensor
        let result = &a + &b;
        assert_eq!(result.get(&[0, 0]), 6.0);

        // Tensor + &Tensor
        let result = a.clone() + &b;
        assert_eq!(result.get(&[0, 0]), 6.0);

        // &Tensor + Tensor
        let result = &a + b.clone();
        assert_eq!(result.get(&[0, 0]), 6.0);

        // Tensor += Tensor
        let mut c = a.clone();
        c += b.clone();
        assert_eq!(c.get(&[0, 0]), 6.0);

        // Tensor += &Tensor
        let mut c = a.clone();
        c += &b;
        assert_eq!(c.get(&[0, 0]), 6.0);
    }

    #[test]
    fn test_tensor_subtraction_operators() {
        let a = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Tensor - Tensor
        let result = a.clone() - b.clone();
        assert_eq!(result.get(&[0, 0]), 4.0);
        assert_eq!(result.get(&[0, 1]), 4.0);
        assert_eq!(result.get(&[1, 0]), 4.0);
        assert_eq!(result.get(&[1, 1]), 4.0);

        // &Tensor - &Tensor
        let result = &a - &b;
        assert_eq!(result.get(&[0, 0]), 4.0);

        // Tensor - &Tensor
        let result = a.clone() - &b;
        assert_eq!(result.get(&[0, 0]), 4.0);

        // &Tensor - Tensor
        let result = &a - b.clone();
        assert_eq!(result.get(&[0, 0]), 4.0);

        // Tensor -= Tensor
        let mut c = a.clone();
        c -= b.clone();
        assert_eq!(c.get(&[0, 0]), 4.0);

        // Tensor -= &Tensor
        let mut c = a.clone();
        c -= &b;
        assert_eq!(c.get(&[0, 0]), 4.0);
    }

    #[test]
    fn test_tensor_multiplication_operators() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        // Tensor * Tensor
        let result = a.clone() * b.clone();
        assert_eq!(result.get(&[0, 0]), 2.0);
        assert_eq!(result.get(&[0, 1]), 6.0);
        assert_eq!(result.get(&[1, 0]), 12.0);
        assert_eq!(result.get(&[1, 1]), 20.0);

        // &Tensor * &Tensor
        let result = &a * &b;
        assert_eq!(result.get(&[0, 0]), 2.0);

        // Tensor * &Tensor
        let result = a.clone() * &b;
        assert_eq!(result.get(&[0, 0]), 2.0);

        // &Tensor * Tensor
        let result = &a * b.clone();
        assert_eq!(result.get(&[0, 0]), 2.0);

        // Tensor *= Tensor
        let mut c = a.clone();
        c *= b.clone();
        assert_eq!(c.get(&[0, 0]), 2.0);

        // Tensor *= &Tensor
        let mut c = a.clone();
        c *= &b;
        assert_eq!(c.get(&[0, 0]), 2.0);
    }

    #[test]
    fn test_tensor_division_operators() {
        let a = Tensor::from_slice(&[10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0, 4.0, 5.0, 8.0], vec![2, 2]).unwrap();

        // Tensor / Tensor
        let result = a.clone() / b.clone();
        assert_eq!(result.get(&[0, 0]), 5.0);
        assert_eq!(result.get(&[0, 1]), 5.0);
        assert_eq!(result.get(&[1, 0]), 6.0);
        assert_eq!(result.get(&[1, 1]), 5.0);

        // &Tensor / &Tensor
        let result = &a / &b;
        assert_eq!(result.get(&[0, 0]), 5.0);

        // Tensor / &Tensor
        let result = a.clone() / &b;
        assert_eq!(result.get(&[0, 0]), 5.0);

        // &Tensor / Tensor
        let result = &a / b.clone();
        assert_eq!(result.get(&[0, 0]), 5.0);

        // Tensor /= Tensor
        let mut c = a.clone();
        c /= b.clone();
        assert_eq!(c.get(&[0, 0]), 5.0);

        // Tensor /= &Tensor
        let mut c = a.clone();
        c /= &b;
        assert_eq!(c.get(&[0, 0]), 5.0);
    }

    #[test]
    fn test_scalar_operations() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Tensor + f32
        let result = a.clone() + 5.0;
        assert_eq!(result.get(&[0, 0]), 6.0);
        assert_eq!(result.get(&[0, 1]), 7.0);
        assert_eq!(result.get(&[1, 0]), 8.0);
        assert_eq!(result.get(&[1, 1]), 9.0);

        // &Tensor + f32
        let result = &a + 5.0;
        assert_eq!(result.get(&[0, 0]), 6.0);

        // f32 + Tensor
        let result = 5.0 + a.clone();
        assert_eq!(result.get(&[0, 0]), 6.0);

        // f32 + &Tensor
        let result = 5.0 + &a;
        assert_eq!(result.get(&[0, 0]), 6.0);

        // Tensor += f32
        let mut b = a.clone();
        b += 5.0;
        assert_eq!(b.get(&[0, 0]), 6.0);

        // Tensor - f32
        let result = a.clone() - 2.0;
        assert_eq!(result.get(&[0, 0]), -1.0);
        assert_eq!(result.get(&[0, 1]), 0.0);
        assert_eq!(result.get(&[1, 0]), 1.0);
        assert_eq!(result.get(&[1, 1]), 2.0);

        // &Tensor - f32
        let result = &a - 2.0;
        assert_eq!(result.get(&[0, 0]), -1.0);

        // f32 - Tensor
        let result = 10.0 - a.clone();
        assert_eq!(result.get(&[0, 0]), 9.0);
        assert_eq!(result.get(&[0, 1]), 8.0);
        assert_eq!(result.get(&[1, 0]), 7.0);
        assert_eq!(result.get(&[1, 1]), 6.0);

        // f32 - &Tensor
        let result = 10.0 - &a;
        assert_eq!(result.get(&[0, 0]), 9.0);

        // Tensor -= f32
        let mut b = a.clone();
        b -= 2.0;
        assert_eq!(b.get(&[0, 0]), -1.0);

        // Tensor * f32
        let result = a.clone() * 3.0;
        assert_eq!(result.get(&[0, 0]), 3.0);
        assert_eq!(result.get(&[0, 1]), 6.0);
        assert_eq!(result.get(&[1, 0]), 9.0);
        assert_eq!(result.get(&[1, 1]), 12.0);

        // &Tensor * f32
        let result = &a * 3.0;
        assert_eq!(result.get(&[0, 0]), 3.0);

        // f32 * Tensor
        let result = 3.0 * a.clone();
        assert_eq!(result.get(&[0, 0]), 3.0);

        // f32 * &Tensor
        let result = 3.0 * &a;
        assert_eq!(result.get(&[0, 0]), 3.0);

        // Tensor *= f32
        let mut b = a.clone();
        b *= 3.0;
        assert_eq!(b.get(&[0, 0]), 3.0);

        // Tensor / f32
        let result = a.clone() / 2.0;
        assert_eq!(result.get(&[0, 0]), 0.5);
        assert_eq!(result.get(&[0, 1]), 1.0);
        assert_eq!(result.get(&[1, 0]), 1.5);
        assert_eq!(result.get(&[1, 1]), 2.0);

        // &Tensor / f32
        let result = &a / 2.0;
        assert_eq!(result.get(&[0, 0]), 0.5);

        // f32 / Tensor
        let result = 10.0 / a.clone();
        assert_eq!(result.get(&[0, 0]), 10.0);
        assert_eq!(result.get(&[0, 1]), 5.0);
        assert!((result.get(&[1, 0]) - (10.0 / 3.0)).abs() < 1e-6);
        assert_eq!(result.get(&[1, 1]), 2.5);

        // f32 / &Tensor
        let result = 10.0 / &a;
        assert_eq!(result.get(&[0, 0]), 10.0);

        // Tensor /= f32
        let mut b = a.clone();
        b /= 2.0;
        assert_eq!(b.get(&[0, 0]), 0.5);
    }

    #[test]
    fn test_negation_operator() {
        let a = Tensor::from_slice(&[1.0, -2.0, 3.0, -4.0], vec![2, 2]).unwrap();

        // -Tensor
        let result = -a.clone();
        assert_eq!(result.get(&[0, 0]), -1.0);
        assert_eq!(result.get(&[0, 1]), 2.0);
        assert_eq!(result.get(&[1, 0]), -3.0);
        assert_eq!(result.get(&[1, 1]), 4.0);

        // -&Tensor
        let result = -&a;
        assert_eq!(result.get(&[0, 0]), -1.0);
    }

    /// Test complex operator chaining and compound expressions
    ///
    /// Verifies that complex mathematical expressions with multiple operators
    /// work correctly, including parentheses and operator precedence.
    #[test]
    fn test_operator_chaining() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0, 3.0, 4.0, 5.0], vec![2, 2]).unwrap();

        // Complex expression: (a + b) * 2 - 1
        let result = (a.clone() + b.clone()) * 2.0 - 1.0;
        assert_eq!(result.get(&[0, 0]), 5.0); // (1+2)*2-1 = 5
        assert_eq!(result.get(&[0, 1]), 9.0); // (2+3)*2-1 = 9
        assert_eq!(result.get(&[1, 0]), 13.0); // (3+4)*2-1 = 13
        assert_eq!(result.get(&[1, 1]), 17.0); // (4+5)*2-1 = 17

        // With references: (&a + &b) * 2 - 1
        let result = (&a + &b) * 2.0 - 1.0;
        assert_eq!(result.get(&[0, 0]), 5.0);
    }
}
