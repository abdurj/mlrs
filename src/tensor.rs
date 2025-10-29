use std::cell::RefCell;
use std::cmp::max;
use std::ops::{Add, Mul, Sub};
use std::os::unix::thread;
use std::rc::Rc;
use std::result;

use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::Rng;

/// Main tensor struct that holds data and gradient information
// #[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    numel: usize,
    pub grad: Option<Vec<f32>>,
    pub requires_grad: bool,
    pub grad_fn: Option<Rc<RefCell<dyn GradFn>>>,
}

/// Trait for gradient functions - each operation implements this
pub trait GradFn {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>>;
}

/// Gradient function for matrix multiplication
pub struct MatMulBackward {
    input1: Tensor,
    input2: Tensor,
}

impl GradFn for MatMulBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // TODO: Implement gradient computation for matmul
        // Hint: grad_input1 = grad_output @ input2.T
        // Hint: grad_input2 = input1.T @ grad_output
        todo!("Implement MatMulBackward")
    }
}

/// Gradient function for addition
pub struct AddBackward;

impl GradFn for AddBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // TODO: Implement gradient computation for addition
        // Hint: gradients flow equally to both inputs
        todo!("Implement AddBackward")
    }
}

/// Gradient function for ReLU activation
pub struct ReLUBackward {
    input: Tensor,
}

impl GradFn for ReLUBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // TODO: Implement gradient computation for ReLU
        // Hint: gradient is 0 where input <= 0, otherwise passes through
        todo!("Implement ReLUBackward")
    }
}

impl Tensor {
    /// Create a new tensor from data and shape
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel = shape.iter().fold(1, |acc, &x| acc * x);
        if numel != data.len() {
            panic!(
                "Invalid shape: {:?} for data of length: {}",
                shape,
                data.len()
            )
        }

        Tensor {
            data,
            shape,
            numel,
            grad: None,
            requires_grad: false,
            grad_fn: None,
        }
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().fold(1, |acc, &x| acc * x);
        let data = vec![0.0; numel];

        Tensor::new(data, shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel = shape.iter().fold(1, |acc, &x| acc * x);
        let data = vec![1.0; numel];

        Tensor::new(data, shape)
    }

    /// Create a tensor with random values between -1 and 1
    pub fn randn(shape: Vec<usize>) -> Self {
        let numel = shape.iter().fold(1, |acc, &x| acc * x);
        let mut data: Vec<f32> = vec![0.0; numel];

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        for i in data.iter_mut() {
            *i = uniform.sample(&mut rng);
        }

        Tensor::new(data, shape)
    }

    /// Builder method to enable gradient tracking
    pub fn requires_grad(mut self, req: bool) -> Self {
        self.requires_grad = req;
        if req {
            self.grad = Some(vec![0.0; self.numel()]);
        }

        self
    }

    /// Zero out all gradients
    pub fn zero_grad(&mut self) {
        if self.requires_grad {
            if let Some(grad) = &mut self.grad {
                for g in grad.iter_mut() {
                    *g = 0.0;
                }
            }
        }
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let numel = new_shape.iter().fold(1, |acc, &x| acc * x);
        assert!(
            numel == self.numel,
            "Reshape error: total number of elements must remain the same."
        );
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            numel: self.numel,
            grad: self.grad.clone(),
            requires_grad: self.requires_grad,
            grad_fn: self.grad_fn.clone(),
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // TODO: Assert both tensors are 2D
        // TODO: Assert inner dimensions match (self.shape[1] == other.shape[0])
        // TODO: Implement matrix multiplication algorithm
        // TODO: If requires_grad, attach MatMulBackward grad_fn
        todo!("Implement matmul")
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: add AddBackward grad_fn if requires_grad
        result
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: Assert shapes match
        // TODO: Zip iterators and subtract element-wise
        result
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: grad fn
        todo!("Implement mul")
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.iter().map(|a| a * scalar).collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: grad fn
        todo!("Implement mul_scalar")
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.max(0.0)).collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: grad fn
        todo!("Implement relu")
    }

    /// Sigmoid activation: 1 / (1 + e^-x)
    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: grad fn
        todo!("Implement sigmoid")
    }

    /// Sum all elements
    pub fn sum(&self) -> Tensor {
        let data = self.data.iter().cloned().sum();
        let mut result = Tensor::new(vec![data], vec![1]);
        // TODO: grad fn
        result
    }

    /// Mean of all elements
    pub fn mean(&self) -> Tensor {
        let sum = self.sum();
        let count = self.numel() as f32;
        let data = sum.data.iter().map(|x| x / count).collect();
        let mut result = Tensor::new(data, vec![1]);
        // TODO: grad fn
        result
    }

    /// Backward pass - propagate gradients
    pub fn backward(&mut self) {
        // TODO: Initialize gradient as ones
        // TODO: Call grad_fn.backward() if it exists
        // Note: Full implementation would do topological sort
        todo!("Implement backward")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_factory_methods() {
        // Test tensor creation
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.data.len(), 4);
        assert_eq!(t.shape, vec![2, 2]);

        // Create zeros
        let t = Tensor::zeros(vec![2, 2]);
        assert_eq!(t.data, vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(t.shape, vec![2, 2]);

        // Create ones
        let t = Tensor::ones(vec![2, 2]);
        assert_eq!(t.data, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(t.shape, vec![2, 2]);

        // Create random
        let t = Tensor::randn(vec![2, 2]);
        assert_eq!(t.data.len(), 4);
        assert_eq!(t.shape, vec![2, 2]);
        assert!(t.data.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    #[should_panic(expected = "Invalid shape")]
    fn test_invalid_tensor_creation() {
        // TODO: Test tensor creation
        let _t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![3, 2]);
    }
}
