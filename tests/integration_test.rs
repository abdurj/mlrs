//! Integration tests to ensure all tensor modules are properly integrated

use rust_nn::Tensor;

#[test]
fn test_tensor_operations_integration() {
    // Test basic tensor operations
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
    let y = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![2, 2]).requires_grad(true);
    
    // Test ops from tensor.rs
    let z = x.add(&y);
    assert_eq!(z.data, vec![3.0, 5.0, 7.0, 9.0]);
    
    let w = x.mul(&y);
    assert_eq!(w.data, vec![2.0, 6.0, 12.0, 20.0]);
    
    // Test matmul (uses kernels/gemm.rs)
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).requires_grad(true);
    let c = a.matmul(&b);
    assert_eq!(c.data, vec![1.0, 2.0, 3.0, 4.0]);
    
    // Test backward pass
    c.backward();
    let a_grad = a.grad.borrow();
    assert!(a_grad.iter().all(|&g| g != 0.0));
}

#[test]
fn test_ops_module_integration() {
    use rust_nn::tensor::ops::softmax;
    
    // Test softmax (from ops.rs) - operates along last axis
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
    let y = softmax(&x);
    
    // Check softmax properties: sum along last axis should be ~1.0 for each row
    let row1_sum = y.data[0] + y.data[1] + y.data[2];
    let row2_sum = y.data[3] + y.data[4] + y.data[5];
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
    
    // Verify that softmax operation was created (proving ops.rs is used)
    assert_eq!(y.shape, vec![2, 3]);
}

#[test]
fn test_layer_norm_integration() {
    use rust_nn::tensor::ops::layer_norm;
    
    // Test layer_norm (from ops.rs) - normalizes along last axis
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
    let y = layer_norm(&x, 1e-5);
    
    // Check that output has approximately zero mean per row
    let row1_mean = (y.data[0] + y.data[1] + y.data[2]) / 3.0;
    let row2_mean = (y.data[3] + y.data[4] + y.data[5]) / 3.0;
    assert!(row1_mean.abs() < 1e-5);
    assert!(row2_mean.abs() < 1e-5);
    
    // Verify that layer_norm operation was created (proving ops.rs is used)
    assert_eq!(y.shape, vec![2, 3]);
}

#[test]
fn test_kernels_module_integration() {
    // Test gemm kernels directly
    use rust_nn::tensor::kernels::{matmul, matmul_backward_left, matmul_backward_right};
    
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let a_shape = vec![2, 3];
    let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_shape = vec![3, 2];
    
    let c = matmul(&a, &a_shape, &b, &b_shape);
    assert_eq!(c.len(), 4); // [2x2] output
    
    // Test backward functions
    let grad_output = vec![1.0, 1.0, 1.0, 1.0];
    let grad_shape = vec![2, 2];
    
    let mut a_grad = vec![0.0; 6];
    matmul_backward_left(&grad_output, &grad_shape, &b, &b_shape, &mut a_grad);
    assert!(a_grad.iter().all(|&g| g != 0.0));
    
    let mut b_grad = vec![0.0; 6];
    matmul_backward_right(&a, &a_shape, &grad_output, &grad_shape, &mut b_grad);
    assert!(b_grad.iter().all(|&g| g != 0.0));
}

#[test]
fn test_reduction_operations_integration() {
    // Test sum_axis and mean_axis (from tensor.rs)
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
    
    // Test sum along axis
    let sum_result = x.sum_axis(Some(0), false);
    assert_eq!(sum_result.shape, vec![3]);
    assert_eq!(sum_result.data, vec![5.0, 7.0, 9.0]);
    
    // Test mean along axis
    let mean_result = x.mean_axis(Some(1), true);
    assert_eq!(mean_result.shape, vec![2, 1]);
    assert_eq!(mean_result.data, vec![2.0, 5.0]);
    
    // Test backward
    sum_result.backward();
    let x_grad = x.grad.borrow();
    assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_activations_integration() {
    // Test various activation functions (from tensor.rs)
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]).requires_grad(true);
    
    // ReLU
    let relu_result = x.relu();
    assert_eq!(relu_result.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    
    // Sigmoid
    let sigmoid_result = x.sigmoid();
    assert!(sigmoid_result.data[2] > 0.49 && sigmoid_result.data[2] < 0.51); // sigmoid(0) ≈ 0.5
    
    // Test backward
    sigmoid_result.backward();
    let x_grad = x.grad.borrow();
    assert!(x_grad.iter().all(|&g| g > 0.0)); // sigmoid gradient is always positive
}

#[test]
fn test_reshape_transpose_integration() {
    // Test reshape and transpose (from tensor.rs)
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
    
    // Reshape
    let reshaped = x.reshape(vec![3, 2]);
    assert_eq!(reshaped.shape, vec![3, 2]);
    assert_eq!(reshaped.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    
    // Transpose
    let transposed = x.transpose();
    assert_eq!(transposed.shape, vec![3, 2]);
    assert_eq!(transposed.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    
    // Test backward
    transposed.backward();
    let x_grad = x.grad.borrow();
    assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn test_end_to_end_computation_graph() {
    use rust_nn::tensor::ops::softmax;
    
    // Complex computation using multiple modules
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
    let w = Tensor::new(vec![0.5, 0.5, 0.5, 0.5], vec![2, 2]).requires_grad(true);
    let b = Tensor::new(vec![0.1, 0.1, 0.1, 0.1], vec![2, 2]).requires_grad(true);
    
    // Forward: y = softmax(relu(x @ w + b))
    let z1 = x.matmul(&w);  // Uses kernels/gemm.rs
    let z2 = z1.add(&b);     // Uses tensor.rs
    let z3 = z2.relu();      // Uses tensor.rs
    let y = softmax(&z3);    // Uses ops.rs
    
    // Check output shape
    assert_eq!(y.shape, vec![2, 2]);
    
    // Check softmax properties: sum along last axis should be ~1.0 per row
    let row1_sum = y.data[0] + y.data[1];
    let row2_sum = y.data[2] + y.data[3];
    assert!((row1_sum - 1.0).abs() < 1e-6);
    assert!((row2_sum - 1.0).abs() < 1e-6);
    
    assert_eq!(y.shape, vec![2, 2]);
    assert!(y.data.iter().all(|&v| (0.0..=1.0).contains(&v))); // softmax outputs probabilities
}
