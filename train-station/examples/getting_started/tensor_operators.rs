//! Tensor Operators Example
//!
//! This example demonstrates Rust operator overloading for tensors in Train Station:
//! - Tensor-tensor operations (+, -, *, /)
//! - Tensor-scalar operations (+, -, *, /)
//! - Assignment operators (+=, -=, *=, /=)
//! - Operator chaining and complex expressions
//! - Broadcasting behavior
//! - Equivalence between operators and method calls
//!
//! # Learning Objectives
//!
//! - Understand how Train Station implements Rust operator overloading
//! - Learn to use natural mathematical expressions with tensors
//! - Explore tensor broadcasting and shape compatibility
//! - Compare operator syntax with explicit method calls
//!
//! # Prerequisites
//!
//! - Basic Rust knowledge
//! - Understanding of tensor basics (see tensor_basics.rs)
//! - Familiarity with operator overloading concepts
//!
//! # Usage
//!
//! ```bash
//! cargo run --example tensor_operators
//! ```

use train_station::Tensor;

fn main() {
    println!("=== Tensor Operators Example ===\n");

    demonstrate_basic_operators();
    demonstrate_scalar_operators();
    demonstrate_operator_assignment();
    demonstrate_operator_chaining();
    demonstrate_broadcasting();
    demonstrate_method_equivalence();

    println!("\n=== Example completed successfully! ===");
}

/// Demonstrate basic tensor-tensor operators
fn demonstrate_basic_operators() {
    println!("--- Basic Tensor-Tensor Operators ---");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    println!("Tensor A: {:?}", a.data());
    println!("Tensor B: {:?}", b.data());

    // Addition
    let c = &a + &b;
    println!("A + B: {:?}", c.data());

    // Subtraction
    let d = &a - &b;
    println!("A - B: {:?}", d.data());

    // Multiplication
    let e = &a * &b;
    println!("A * B: {:?}", e.data());

    // Division
    let f = &a / &b;
    println!("A / B: {:?}", f.data());
}

/// Demonstrate tensor-scalar operators
fn demonstrate_scalar_operators() {
    println!("\n--- Tensor-Scalar Operators ---");

    let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    println!("Original tensor: {:?}", tensor.data());

    // Tensor + scalar
    let result1 = &tensor + 5.0;
    println!("Tensor + 5.0: {:?}", result1.data());

    // Scalar + tensor
    let result2 = 5.0 + &tensor;
    println!("5.0 + Tensor: {:?}", result2.data());

    // Tensor - scalar
    let result3 = &tensor - 2.0;
    println!("Tensor - 2.0: {:?}", result3.data());

    // Tensor * scalar
    let result4 = &tensor * 3.0;
    println!("Tensor * 3.0: {:?}", result4.data());

    // Scalar * tensor
    let result5 = 3.0 * &tensor;
    println!("3.0 * Tensor: {:?}", result5.data());

    // Tensor / scalar
    let result6 = &tensor / 2.0;
    println!("Tensor / 2.0: {:?}", result6.data());
}

/// Demonstrate assignment operators
fn demonstrate_operator_assignment() {
    println!("\n--- Assignment Operators ---");

    let mut tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    println!("Original tensor: {:?}", tensor.data());

    // In-place addition
    tensor += 5.0;
    println!("After += 5.0: {:?}", tensor.data());

    // In-place subtraction
    tensor -= 2.0;
    println!("After -= 2.0: {:?}", tensor.data());

    // In-place multiplication
    tensor *= 3.0;
    println!("After *= 3.0: {:?}", tensor.data());

    // In-place division
    tensor /= 2.0;
    println!("After /= 2.0: {:?}", tensor.data());
}

/// Demonstrate operator chaining and complex expressions
fn demonstrate_operator_chaining() {
    println!("\n--- Operator Chaining ---");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();
    let c = Tensor::from_slice(&[9.0, 10.0, 11.0, 12.0], vec![2, 2]).unwrap();

    println!("Tensor A: {:?}", a.data());
    println!("Tensor B: {:?}", b.data());
    println!("Tensor C: {:?}", c.data());

    // Complex expression: (A + B) * C - 5
    let result = (&a + &b) * &c - 5.0;
    println!("(A + B) * C - 5: {:?}", result.data());

    // Another complex expression: A * 2 + B / 2
    let result2 = &a * 2.0 + &b / 2.0;
    println!("A * 2 + B / 2: {:?}", result2.data());

    // Negation and addition: -A + B * C
    let result3 = -&a + &b * &c;
    println!("-A + B * C: {:?}", result3.data());

    // Division with parentheses: (A + B) / (C - 1)
    let result4 = (&a + &b) / (&c - 1.0);
    println!("(A + B) / (C - 1): {:?}", result4.data());
}

/// Demonstrate broadcasting behavior
fn demonstrate_broadcasting() {
    println!("\n--- Broadcasting ---");

    // 2D tensor
    let tensor_2d = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    println!(
        "2D tensor: shape {:?}, data: {:?}",
        tensor_2d.shape().dims,
        tensor_2d.data()
    );

    // 1D tensor (will be broadcasted)
    let tensor_1d = Tensor::from_slice(&[10.0, 20.0], vec![2]).unwrap();
    println!(
        "1D tensor: shape {:?}, data: {:?}",
        tensor_1d.shape().dims,
        tensor_1d.data()
    );

    // Broadcasting addition
    let broadcast_sum = &tensor_2d + &tensor_1d;
    println!(
        "Broadcast sum: shape {:?}, data: {:?}",
        broadcast_sum.shape().dims,
        broadcast_sum.data()
    );

    // Broadcasting multiplication
    let broadcast_mul = &tensor_2d * &tensor_1d;
    println!(
        "Broadcast multiplication: shape {:?}, data: {:?}",
        broadcast_mul.shape().dims,
        broadcast_mul.data()
    );

    // Broadcasting with scalar
    let broadcast_scalar = &tensor_2d + 100.0;
    println!(
        "Broadcast scalar: shape {:?}, data: {:?}",
        broadcast_scalar.shape().dims,
        broadcast_scalar.data()
    );
}

/// Demonstrate equivalence between operators and method calls
fn demonstrate_method_equivalence() {
    println!("\n--- Operator vs Method Call Equivalence ---");

    let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
    let b = Tensor::from_slice(&[5.0, 6.0, 7.0, 8.0], vec![2, 2]).unwrap();

    // Addition: operator vs method
    let operator_result = &a + &b;
    let method_result = a.add_tensor(&b);

    println!("A + B (operator): {:?}", operator_result.data());
    println!("A.add_tensor(B): {:?}", method_result.data());
    println!(
        "Results are equal: {}",
        operator_result.data() == method_result.data()
    );

    // Multiplication: operator vs method
    let operator_result = &a * &b;
    let method_result = a.mul_tensor(&b);

    println!("A * B (operator): {:?}", operator_result.data());
    println!("A.mul_tensor(B): {:?}", method_result.data());
    println!(
        "Results are equal: {}",
        operator_result.data() == method_result.data()
    );

    // Scalar addition: operator vs method
    let operator_result = &a + 5.0;
    let method_result = a.add_scalar(5.0);

    println!("A + 5.0 (operator): {:?}", operator_result.data());
    println!("A.add_scalar(5.0): {:?}", method_result.data());
    println!(
        "Results are equal: {}",
        operator_result.data() == method_result.data()
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operators() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();

        let sum = &a + &b;
        assert_eq!(sum.data(), &[4.0, 6.0]);

        let product = &a * &b;
        assert_eq!(product.data(), &[3.0, 8.0]);
    }

    #[test]
    fn test_scalar_operators() {
        let tensor = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();

        let result = &tensor + 5.0;
        assert_eq!(result.data(), &[6.0, 7.0]);

        let result = 5.0 + &tensor;
        assert_eq!(result.data(), &[6.0, 7.0]);
    }

    #[test]
    fn test_assignment_operators() {
        let mut tensor = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();

        tensor += 3.0;
        assert_eq!(tensor.data(), &[4.0, 5.0]);

        tensor *= 2.0;
        assert_eq!(tensor.data(), &[8.0, 10.0]);
    }

    #[test]
    fn test_operator_chaining() {
        let a = Tensor::from_slice(&[1.0, 2.0], vec![2]).unwrap();
        let b = Tensor::from_slice(&[3.0, 4.0], vec![2]).unwrap();

        let result = (&a + &b) * 2.0;
        assert_eq!(result.data(), &[8.0, 12.0]);
    }
}
