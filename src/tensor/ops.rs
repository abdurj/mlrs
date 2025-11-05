//! Additional tensor operations
//!
//! This module contains higher-level tensor operations like softmax,
//! layer normalization, and other common neural network operations.

use super::Tensor;
use std::rc::Rc;

/// Softmax activation along the last axis
///
/// Computes: softmax(x)_i = exp(x_i) / sum(exp(x_j))
///
/// For numerical stability, we subtract the max value before exponentiating:
/// softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
pub fn softmax(input: &Tensor) -> Tensor {
    assert!(
        !input.shape.is_empty(),
        "Cannot compute softmax on scalar tensor"
    );

    let axis = input.shape.len() - 1;
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };

    let mut data = vec![0.0; input.numel()];

    // Process each vector along the last axis
    for outer in 0..outer_size {
        let base_idx = outer * axis_size;

        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        for i in 0..axis_size {
            max_val = max_val.max(input.data[base_idx + i]);
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for i in 0..axis_size {
            let exp_val = (input.data[base_idx + i] - max_val).exp();
            data[base_idx + i] = exp_val;
            sum += exp_val;
        }

        // Normalize by sum
        for i in 0..axis_size {
            data[base_idx + i] /= sum;
        }
    }

    let result = Tensor::new(data, input.shape.clone());

    if input.requires_grad {
        let input_grad = Rc::clone(&input.grad);
        let result_grad = Rc::clone(&result.grad);
        let result_data = result.data.clone();
        let shape = input.shape.clone();

        Tensor::with_grad(
            result,
            vec![input],
            Box::new(move || {
                let grad_output = result_grad.borrow();
                let mut input_g = input_grad.borrow_mut();

                let axis = shape.len() - 1;
                let axis_size = shape[axis];
                let outer_size: usize = shape[..axis].iter().product();
                let outer_size = if outer_size == 0 { 1 } else { outer_size };

                // Jacobian of softmax: ∂y_i/∂x_j = y_i * (δ_ij - y_j)
                // Gradient: ∂L/∂x_i = Σ_j (∂L/∂y_j * ∂y_j/∂x_i)
                //                    = Σ_j (grad_j * y_j * (δ_ij - y_i))
                //                    = grad_i * y_i - y_i * Σ_j(grad_j * y_j)
                for outer in 0..outer_size {
                    let base_idx = outer * axis_size;

                    // Compute dot product: sum(grad * softmax_output)
                    let mut dot_product = 0.0;
                    for i in 0..axis_size {
                        dot_product += grad_output[base_idx + i] * result_data[base_idx + i];
                    }

                    // Apply gradient
                    for i in 0..axis_size {
                        let idx = base_idx + i;
                        input_g[idx] += result_data[idx] * (grad_output[idx] - dot_product);
                    }
                }
            }),
        )
    } else {
        result
    }
}

/// Layer normalization
///
/// Normalizes the input along the last axis to have mean 0 and variance 1.
///
/// # Arguments
/// * `input` - The input tensor
/// * `eps` - Small constant for numerical stability (default: 1e-5)
pub fn layer_norm(input: &Tensor, eps: f32) -> Tensor {
    assert!(
        !input.shape.is_empty(),
        "Cannot compute layer norm on scalar tensor"
    );

    let axis = input.shape.len() - 1;
    let axis_size = input.shape[axis];
    let outer_size: usize = input.shape[..axis].iter().product();
    let outer_size = if outer_size == 0 { 1 } else { outer_size };

    let mut data = vec![0.0; input.numel()];

    // Compute mean and variance for each vector
    for outer in 0..outer_size {
        let base_idx = outer * axis_size;

        // Compute mean
        let mut sum = 0.0;
        for i in 0..axis_size {
            sum += input.data[base_idx + i];
        }
        let mean = sum / axis_size as f32;

        // Compute variance
        let mut var_sum = 0.0;
        for i in 0..axis_size {
            let diff = input.data[base_idx + i] - mean;
            var_sum += diff * diff;
        }
        let variance = var_sum / axis_size as f32;

        // Normalize
        let std = (variance + eps).sqrt();
        for i in 0..axis_size {
            data[base_idx + i] = (input.data[base_idx + i] - mean) / std;
        }
    }

    let result = Tensor::new(data, input.shape.clone());

    if input.requires_grad {
        let input_grad = Rc::clone(&input.grad);
        let result_grad = Rc::clone(&result.grad);
        let result_data = result.data.clone();
        let shape = input.shape.clone();
        let input_data = input.data.clone();

        Tensor::with_grad(
            result,
            vec![input],
            Box::new(move || {
                let grad_output = result_grad.borrow();
                let mut input_g = input_grad.borrow_mut();

                let axis = shape.len() - 1;
                let axis_size = shape[axis];
                let outer_size: usize = shape[..axis].iter().product();
                let outer_size = if outer_size == 0 { 1 } else { outer_size };

                // Recompute mean and variance (needed for backward pass)
                for outer in 0..outer_size {
                    let base_idx = outer * axis_size;

                    // Compute mean
                    let mut sum = 0.0;
                    for i in 0..axis_size {
                        sum += input_data[base_idx + i];
                    }
                    let mean = sum / axis_size as f32;

                    // Compute variance
                    let mut var_sum = 0.0;
                    for i in 0..axis_size {
                        let diff = input_data[base_idx + i] - mean;
                        var_sum += diff * diff;
                    }
                    let variance = var_sum / axis_size as f32;
                    let std = (variance + eps).sqrt();

                    // Compute gradient
                    // dL/dx = (1/std) * (dL/dy - mean(dL/dy) - normalized_x * mean(dL/dy * normalized_x))
                    let mut grad_sum = 0.0;
                    let mut grad_dot = 0.0;
                    for i in 0..axis_size {
                        let idx = base_idx + i;
                        grad_sum += grad_output[idx];
                        grad_dot += grad_output[idx] * result_data[idx];
                    }
                    let grad_mean = grad_sum / axis_size as f32;
                    let grad_dot_mean = grad_dot / axis_size as f32;

                    for i in 0..axis_size {
                        let idx = base_idx + i;
                        input_g[idx] +=
                            (grad_output[idx] - grad_mean - result_data[idx] * grad_dot_mean) / std;
                    }
                }
            }),
        )
    } else {
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    #[test]
    fn test_softmax_forward() {
        // Test softmax on a simple vector
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let y = softmax(&x);

        // Check that output sums to 1
        let sum: f32 = y.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        assert!(y.data.iter().all(|&v| v > 0.0));

        // Check that larger inputs produce larger outputs
        assert!(y.data[2] > y.data[1]);
        assert!(y.data[1] > y.data[0]);

        println!("✓ Softmax forward test passed!");
    }

    #[test]
    fn test_softmax_backward() {
        // Test softmax gradient
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let y = softmax(&x);

        y.backward();

        let x_grad = x.grad.borrow();
        // Gradient should sum to 0 (property of softmax)
        let grad_sum: f32 = x_grad.iter().sum();
        assert!(grad_sum.abs() < 1e-6);

        println!("✓ Softmax backward test passed!");
        println!("  Gradients: {:?}", *x_grad);
    }

    #[test]
    fn test_softmax_2d() {
        // Test softmax on 2D tensor (batch of vectors)
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = softmax(&x);

        // Check each row sums to 1
        let row1_sum: f32 = y.data[0..3].iter().sum();
        let row2_sum: f32 = y.data[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);

        println!("✓ Softmax 2D test passed!");
    }

    #[test]
    fn test_layer_norm_forward() {
        // Test layer norm on a simple vector
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let y = layer_norm(&x, 1e-5);

        // Check mean is approximately 0
        let mean: f32 = y.data.iter().sum::<f32>() / y.data.len() as f32;
        assert!(mean.abs() < 1e-5);

        // Check variance is approximately 1
        let variance: f32 = y.data.iter().map(|&v| v * v).sum::<f32>() / y.data.len() as f32;
        assert!((variance - 1.0).abs() < 1e-4);

        println!("✓ Layer norm forward test passed!");
    }

    #[test]
    fn test_layer_norm_backward() {
        // Test layer norm gradient
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);
        let y = layer_norm(&x, 1e-5);

        y.backward();

        let x_grad = x.grad.borrow();
        // Gradients should exist
        assert!(x_grad.iter().all(|&g| !g.is_nan()));

        println!("✓ Layer norm backward test passed!");
        println!("  Gradients: {:?}", *x_grad);
    }

    #[test]
    fn test_layer_norm_2d() {
        // Test layer norm on 2D tensor
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = layer_norm(&x, 1e-5);

        // Check each row has mean ~0 and variance ~1
        let row1_mean: f32 = y.data[0..3].iter().sum::<f32>() / 3.0;
        let row2_mean: f32 = y.data[3..6].iter().sum::<f32>() / 3.0;
        assert!(row1_mean.abs() < 1e-5);
        assert!(row2_mean.abs() < 1e-5);

        println!("✓ Layer norm 2D test passed!");
    }
}
