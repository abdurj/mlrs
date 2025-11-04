use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::{Rc, Weak};

use rand::distributions::Uniform;
use rand::prelude::Distribution;

/// Main tensor struct that holds data and gradient information
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    numel: usize,
    // Use RefCell for interior mutability of gradients
    pub grad: Rc<RefCell<Vec<f32>>>,
    pub requires_grad: bool,
    // Graph node for backward pass (only created when requires_grad is true)
    graph_node: Option<Rc<GraphNode>>,
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            data: self.data.clone(),
            shape: self.shape.clone(),
            numel: self.numel,
            grad: Rc::clone(&self.grad),
            requires_grad: self.requires_grad,
            graph_node: self.graph_node.clone(),
        }
    }
}

/// Type alias for the backward function closure
/// Takes no arguments, returns nothing - it directly mutates gradients via captured references
type BackwardFn = Box<dyn Fn()>;

/// GraphNode represents a node in the computation graph
/// Separates graph structure from tensor data
pub struct GraphNode {
    // Shared reference to gradient storage
    grad: Rc<RefCell<Vec<f32>>>,
    // Backward function for this operation
    backward_fn: BackwardFn,
    // Parent nodes in the computation graph (Weak to avoid cycles)
    prev: Vec<Weak<GraphNode>>,
}

impl Tensor {
    /// Create a new tensor from data and shape
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
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
            grad: Rc::new(RefCell::new(vec![0.0; numel])),
            requires_grad: false,
            graph_node: None,
        }
    }

    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Helper to attach gradient tracking to a result tensor
    /// Takes parent tensors and a backward closure, sets up the computation graph
    pub(super) fn with_grad(
        mut result: Tensor,
        parents: Vec<&Tensor>,
        backward_fn: BackwardFn,
    ) -> Tensor {
        result.requires_grad = true;

        // Collect parent graph nodes (create Weak refs to avoid cycles)
        let parent_nodes: Vec<Weak<GraphNode>> = parents
            .iter()
            .filter_map(|p| p.graph_node.as_ref().map(|n| Rc::downgrade(n)))
            .collect();

        // Create graph node for this result
        let node = Rc::new(GraphNode {
            grad: Rc::clone(&result.grad),
            backward_fn,
            prev: parent_nodes,
        });

        result.graph_node = Some(node);
        result
    }

    /// Helper for binary operations with gradient tracking
    /// Takes two tensors, result data, and a closure that computes gradients
    fn binary_op<F>(&self, other: &Tensor, result_data: Vec<f32>, grad_fn: F) -> Tensor
    where
        F: Fn(&[f32], &mut [f32], &mut [f32]) + 'static,
    {
        let result = Tensor::new(result_data, self.shape.clone());

        if self.requires_grad || other.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let other_grad = Rc::clone(&other.grad);
            let result_grad = Rc::clone(&result.grad);

            // Check if self and other share the same gradient (e.g., x.add(&x))
            let same_tensor = Rc::ptr_eq(&self_grad, &other_grad);

            Self::with_grad(
                result,
                vec![self, other],
                Box::new(move || {
                    let grad_output = result_grad.borrow();

                    if same_tensor {
                        // Both inputs are the same tensor - accumulate once
                        let mut grad = self_grad.borrow_mut();
                        let mut temp_grad = vec![0.0; grad.len()];
                        grad_fn(&grad_output, &mut grad, &mut temp_grad);
                        // Add the "other" gradients to self (since they're the same)
                        for i in 0..grad.len() {
                            grad[i] += temp_grad[i];
                        }
                    } else {
                        // Different tensors - borrow both mutably
                        let mut self_g = self_grad.borrow_mut();
                        let mut other_g = other_grad.borrow_mut();
                        grad_fn(&grad_output, &mut self_g, &mut other_g);
                    }
                }),
            )
        } else {
            result
        }
    }

    /// Create a tensor filled with zeros
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![0.0; numel];

        Tensor::new(data, shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
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
        self
    }

    /// Zero out all gradients
    pub fn zero_grad(&mut self) {
        for g in self.grad.borrow_mut().iter_mut() {
            *g = 0.0;
        }
    }

    /// Reshape tensor to new shape
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let numel = new_shape.iter().fold(1, |acc, &x| acc * x);
        assert!(
            numel == self.numel,
            "Reshape error: total number of elements must remain the same."
        );

        let result = Tensor::new(self.data.clone(), new_shape);

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);

            Self::with_grad(
                result,
                vec![self],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    // Gradient of reshape is just copying values (same memory layout)
                    for i in 0..grad_output.len() {
                        self_g[i] += grad_output[i];
                    }
                }),
            )
        } else {
            result
        }
    }

    /// Matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Use the gemm kernel for the actual matrix multiplication
        let data = super::kernels::gemm::matmul(&self.data, &self.shape, &other.data, &other.shape);
        let result = Tensor::new(data, vec![self.shape[0], other.shape[1]]);

        if self.requires_grad || other.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let other_grad = Rc::clone(&other.grad);
            let result_grad = Rc::clone(&result.grad);

            // TODO: Implement self-matmul (x.matmul(&x)) support
            assert!(
                !Rc::ptr_eq(&self_grad, &other_grad),
                "Self-matmul (x.matmul(&x)) is not yet supported"
            );

            // Clone data and shapes for backward pass
            let self_data = self.data.clone();
            let self_shape = self.shape.clone();
            let other_data = other.data.clone();
            let other_shape = other.shape.clone();

            Self::with_grad(
                result,
                vec![self, other],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let grad_shape = vec![self_shape[0], other_shape[1]];

                    let mut self_g = self_grad.borrow_mut();
                    let mut other_g = other_grad.borrow_mut();

                    // dL/dA = grad_output @ B^T
                    super::kernels::gemm::matmul_backward_left(
                        &grad_output,
                        &grad_shape,
                        &other_data,
                        &other_shape,
                        &mut self_g,
                    );

                    // dL/dB = A^T @ grad_output
                    super::kernels::gemm::matmul_backward_right(
                        &self_data,
                        &self_shape,
                        &grad_output,
                        &grad_shape,
                        &mut other_g,
                    );
                }),
            )
        } else {
            result
        }
    }

    /// Element-wise addition
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();

        self.binary_op(other, data, |grad_output, self_grad, other_grad| {
            for i in 0..grad_output.len() {
                self_grad[i] += grad_output[i];
                other_grad[i] += grad_output[i];
            }
        })
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();

        self.binary_op(other, data, |grad_output, self_grad, other_grad| {
            for i in 0..grad_output.len() {
                self_grad[i] += grad_output[i];
                other_grad[i] -= grad_output[i]; // Note: negative for subtraction
            }
        })
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a * b)
            .collect();

        // Capture input values for gradient computation
        let self_data = self.data.clone();
        let other_data = other.data.clone();

        self.binary_op(other, data, move |grad_output, self_grad, other_grad| {
            for i in 0..grad_output.len() {
                self_grad[i] += grad_output[i] * other_data[i]; // d/dx(x*y) = y
                other_grad[i] += grad_output[i] * self_data[i]; // d/dy(x*y) = x
            }
        })
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|a| a * scalar).collect();
        let result = Tensor::new(data, self.shape.clone());

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);

            Self::with_grad(
                result,
                vec![self],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    for i in 0..grad_output.len() {
                        self_g[i] += grad_output[i] * scalar; // d/dx(x*c) = c
                    }
                }),
            )
        } else {
            result
        }
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Tensor {
        let data = self.data.iter().map(|x| x.max(0.0)).collect();
        let result = Tensor::new(data, self.shape.clone());

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);
            let input_data = self.data.clone(); // Clone Vec<f32> for the mask

            Self::with_grad(
                result,
                vec![self],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    for i in 0..grad_output.len() {
                        if input_data[i] > 0.0 {
                            self_g[i] += grad_output[i];
                        }
                    }
                }),
            )
        } else {
            result
        }
    }

    /// Sigmoid activation: 1 / (1 + e^-x)
    pub fn sigmoid(&self) -> Tensor {
        let data = self.data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let result = Tensor::new(data, self.shape.clone());

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);
            let result_data = result.data.clone(); // Clone Vec<f32> for the mask

            Self::with_grad(
                result,
                vec![self],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    self_g
                        .iter_mut()
                        .zip(grad_output.iter())
                        .zip(result_data.iter())
                        .for_each(|((self_grad, grad_output), result_data)| {
                            *self_grad += result_data * (1.0 - result_data) * grad_output;
                        });
                }),
            )
        } else {
            result
        }
    }

    /// Sum all elements
    pub fn sum(&self) -> Tensor {
        self.sum_axis(None, false)
    }

    /// Sum along a specific axis
    ///
    /// # Arguments
    /// * `axis` - The axis to sum along (None means sum all elements)
    /// * `keepdims` - If true, keep the reduced dimension as size 1
    pub fn sum_axis(&self, axis: Option<usize>, keepdims: bool) -> Tensor {
        match axis {
            None => {
                // Sum all elements
                let data = self.data.iter().cloned().sum();
                let result = Tensor::new(vec![data], vec![1]);

                if self.requires_grad {
                    let self_grad = Rc::clone(&self.grad);
                    let result_grad = Rc::clone(&result.grad);

                    Self::with_grad(
                        result,
                        vec![self],
                        Box::new(move || {
                            let grad_output = result_grad.borrow();
                            let mut self_g = self_grad.borrow_mut();

                            for i in 0..self_g.len() {
                                self_g[i] += grad_output[0]; // Broadcast sum gradient
                            }
                        }),
                    )
                } else {
                    result
                }
            }
            Some(ax) => {
                assert!(
                    ax < self.shape.len(),
                    "Axis {} out of bounds for tensor with {} dimensions",
                    ax,
                    self.shape.len()
                );

                // Calculate output shape
                let mut out_shape = Vec::new();
                for (i, &dim) in self.shape.iter().enumerate() {
                    if i == ax {
                        if keepdims {
                            out_shape.push(1);
                        }
                    } else {
                        out_shape.push(dim);
                    }
                }
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                let out_numel = out_shape.iter().product();
                let mut data = vec![0.0; out_numel];

                // Calculate strides
                let mut strides = vec![1; self.shape.len()];
                for i in (0..self.shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * self.shape[i + 1];
                }

                // Sum along axis
                let axis_size = self.shape[ax];
                let outer_size: usize = self.shape[..ax].iter().product();
                let inner_size: usize = self.shape[ax + 1..].iter().product();

                for outer in 0..outer_size {
                    for inner in 0..inner_size {
                        let mut sum = 0.0;
                        for axis_idx in 0..axis_size {
                            let idx =
                                outer * axis_size * inner_size + axis_idx * inner_size + inner;
                            sum += self.data[idx];
                        }
                        let out_idx = outer * inner_size + inner;
                        data[out_idx] = sum;
                    }
                }

                let result = Tensor::new(data, out_shape.clone());

                if self.requires_grad {
                    let self_grad = Rc::clone(&self.grad);
                    let result_grad = Rc::clone(&result.grad);
                    let self_shape = self.shape.clone();

                    Self::with_grad(
                        result,
                        vec![self],
                        Box::new(move || {
                            let grad_output = result_grad.borrow();
                            let mut self_g = self_grad.borrow_mut();

                            // Broadcast gradient back
                            let axis_size = self_shape[ax];
                            let outer_size: usize = self_shape[..ax].iter().product();
                            let inner_size: usize = self_shape[ax + 1..].iter().product();

                            for outer in 0..outer_size {
                                for inner in 0..inner_size {
                                    let out_idx = outer * inner_size + inner;
                                    let grad_val = grad_output[out_idx];

                                    for axis_idx in 0..axis_size {
                                        let idx = outer * axis_size * inner_size
                                            + axis_idx * inner_size
                                            + inner;
                                        self_g[idx] += grad_val;
                                    }
                                }
                            }
                        }),
                    )
                } else {
                    result
                }
            }
        }
    }

    /// Mean of all elements
    pub fn mean(&self) -> Tensor {
        self.mean_axis(None, false)
    }

    /// Mean along a specific axis
    ///
    /// # Arguments
    /// * `axis` - The axis to take mean along (None means mean of all elements)
    /// * `keepdims` - If true, keep the reduced dimension as size 1
    pub fn mean_axis(&self, axis: Option<usize>, keepdims: bool) -> Tensor {
        match axis {
            None => {
                // Mean of all elements
                let sum = self.sum();
                let count = self.numel() as f32;
                let data = sum.data.iter().map(|x| x / count).collect();
                let result = Tensor::new(data, vec![1]);

                if self.requires_grad {
                    let self_grad = Rc::clone(&self.grad);
                    let result_grad = Rc::clone(&result.grad);

                    Self::with_grad(
                        result,
                        vec![self],
                        Box::new(move || {
                            let grad_output = result_grad.borrow();
                            let mut self_g = self_grad.borrow_mut();

                            for i in 0..self_g.len() {
                                self_g[i] += grad_output[0] / count; // Broadcast mean gradient
                            }
                        }),
                    )
                } else {
                    result
                }
            }
            Some(ax) => {
                assert!(
                    ax < self.shape.len(),
                    "Axis {} out of bounds for tensor with {} dimensions",
                    ax,
                    self.shape.len()
                );

                let sum = self.sum_axis(Some(ax), keepdims);
                let count = self.shape[ax] as f32;
                let data = sum.data.iter().map(|x| x / count).collect();
                let result = Tensor::new(data, sum.shape.clone());

                if self.requires_grad {
                    let self_grad = Rc::clone(&self.grad);
                    let result_grad = Rc::clone(&result.grad);
                    let self_shape = self.shape.clone();

                    Self::with_grad(
                        result,
                        vec![self],
                        Box::new(move || {
                            let grad_output = result_grad.borrow();
                            let mut self_g = self_grad.borrow_mut();

                            // Broadcast gradient back, divided by count
                            let axis_size = self_shape[ax];
                            let outer_size: usize = self_shape[..ax].iter().product();
                            let inner_size: usize = self_shape[ax + 1..].iter().product();

                            for outer in 0..outer_size {
                                for inner in 0..inner_size {
                                    let out_idx = outer * inner_size + inner;
                                    let grad_val = grad_output[out_idx] / count;

                                    for axis_idx in 0..axis_size {
                                        let idx = outer * axis_size * inner_size
                                            + axis_idx * inner_size
                                            + inner;
                                        self_g[idx] += grad_val;
                                    }
                                }
                            }
                        }),
                    )
                } else {
                    result
                }
            }
        }
    }

    /// Transpose a 2D tensor
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.shape.len(), 2, "Transpose only works for 2D tensors");
        let rows = self.shape[0];
        let cols = self.shape[1];
        let mut data = vec![0.0; self.numel()];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
            }
        }

        let result = Tensor::new(data, vec![cols, rows]);

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);

            Self::with_grad(
                result,
                vec![self],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    // Gradient of transpose is transpose of gradient
                    // grad_output has shape [cols, rows], need to transpose back to [rows, cols]
                    for i in 0..cols {
                        for j in 0..rows {
                            self_g[j * cols + i] += grad_output[i * rows + j];
                        }
                    }
                }),
            )
        } else {
            result
        }
    }

    /// Backward pass - propagate gradients
    pub fn backward(&self) {
        // Initialize this tensor's gradient to 1.0 (assuming scalar output)
        {
            let mut grad = self.grad.borrow_mut();
            for g in grad.iter_mut() {
                *g = 1.0;
            }
        }

        // Get the graph node for this tensor
        let root = match &self.graph_node {
            Some(node) => Rc::clone(node),
            None => return, // No graph node means no gradients to compute
        };

        // Build topological sort using DFS on GraphNodes
        let mut topo: Vec<Rc<GraphNode>> = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(
            v: &Rc<GraphNode>,
            visited: &mut HashSet<*const GraphNode>,
            topo: &mut Vec<Rc<GraphNode>>,
        ) {
            let ptr = Rc::as_ptr(v);
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                // Upgrade Weak references to Rc for traversal
                for weak_child in &v.prev {
                    let child = weak_child
                        .upgrade()
                        .expect("Parent in graph dropped unexpectedly");
                    build_topo(&child, visited, topo);
                }
                topo.push(Rc::clone(v));
            }
        }

        build_topo(&root, &mut visited, &mut topo);

        // Call backward functions in reverse topological order
        for node in topo.iter().rev() {
            (node.backward_fn)();
        }
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
        let _t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![3, 2]);
    }

    #[test]
    fn test_backward_simple_add() {
        // Test: z = x + y, all scalars
        // dz/dx = 1, dz/dy = 1
        let x = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        let z = x.add(&y);

        // Before backward, gradients should be zero
        assert_eq!(*x.grad.borrow(), vec![0.0]);
        assert_eq!(*y.grad.borrow(), vec![0.0]);

        // Call backward
        z.backward();

        // After backward, gradients should be 1.0 (gradient flows equally)
        assert_eq!(*x.grad.borrow(), vec![1.0]);
        assert_eq!(*y.grad.borrow(), vec![1.0]);
    }

    #[test]
    fn test_backward_chain() {
        // Test: z = (x + y) + x
        // This tests that gradients accumulate properly
        // dz/dx = 2 (appears twice in computation)
        // dz/dy = 1
        let x = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![3.0], vec![1]).requires_grad(true);

        let temp = x.add(&y); // temp = x + y
        let z = temp.add(&x); // z = temp + x = (x + y) + x

        z.backward();

        // x appears twice, so gradient should be 2.0
        assert_eq!(*x.grad.borrow(), vec![2.0]);
        // y appears once, so gradient should be 1.0
        assert_eq!(*y.grad.borrow(), vec![1.0]);
    }

    #[test]
    fn test_backward_relu() {
        // Test: z = relu(x)
        // relu gradient: 1 where x > 0, else 0
        let x = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).requires_grad(true);
        let z = x.relu();

        z.backward();

        // Gradient should be 0 for x <= 0, and 1 for x > 0
        let x_grad = x.grad.borrow();
        assert_eq!(x_grad[0], 0.0); // x = -1.0, gradient = 0
        assert_eq!(x_grad[1], 0.0); // x = 0.0, gradient = 0
        assert_eq!(x_grad[2], 1.0); // x = 1.0, gradient = 1
        assert_eq!(x_grad[3], 1.0); // x = 2.0, gradient = 1
    }

    #[test]
    fn test_backward_multiple_operations() {
        // Test: z = relu(x + y)
        // Tests composition of operations
        let x = Tensor::new(vec![-1.0, 2.0], vec![2]).requires_grad(true);
        let y = Tensor::new(vec![0.5, -1.0], vec![2]).requires_grad(true);

        let sum = x.add(&y); // [-0.5, 1.0]
        let z = sum.relu(); // [0.0, 1.0]

        z.backward();

        let x_grad = x.grad.borrow();
        let y_grad = y.grad.borrow();

        // First element: -1.0 + 0.5 = -0.5, relu -> 0, so gradient = 0
        assert_eq!(x_grad[0], 0.0);
        assert_eq!(y_grad[0], 0.0);

        // Second element: 2.0 + (-1.0) = 1.0, relu -> 1.0, so gradient = 1
        assert_eq!(x_grad[1], 1.0);
        assert_eq!(y_grad[1], 1.0);
    }

    #[test]
    fn test_gradient_accumulation() {
        // Test that gradients accumulate correctly across multiple backward passes
        let mut x = Tensor::new(vec![1.0], vec![1]).requires_grad(true);

        // First operation
        let y1 = x.add(&x); // y1 = 2x
        y1.backward();

        // Check gradient after first backward
        let grad_after_first = x.grad.borrow().clone();
        assert_eq!(grad_after_first, vec![2.0]);

        // Zero out gradients
        x.zero_grad();
        assert_eq!(*x.grad.borrow(), vec![0.0]);

        // Second operation
        let y2 = x.add(&x);
        y2.backward();

        // Should be same as before (gradients were zeroed)
        assert_eq!(*x.grad.borrow(), vec![2.0]);
    }

    #[test]
    fn test_binary_ops_forward_and_gradients() {
        // Test all binary operations: add, sub, mul with forward and backward passes
        let mut x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let mut y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);

        // Test addition: z = x + y
        let z_add = x.add(&y);
        assert_eq!(z_add.data, vec![3.0, 5.0, 7.0]);
        z_add.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0]); // dz/dx = 1
        assert_eq!(*y.grad.borrow(), vec![1.0, 1.0, 1.0]); // dz/dy = 1

        // Reset gradients
        x.zero_grad();
        y.zero_grad();

        // Test subtraction: z = x - y
        let z_sub = x.sub(&y);
        assert_eq!(z_sub.data, vec![1.0, 1.0, 1.0]);
        z_sub.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0]); // dz/dx = 1
        assert_eq!(*y.grad.borrow(), vec![-1.0, -1.0, -1.0]); // dz/dy = -1

        // Reset gradients
        x.zero_grad();
        y.zero_grad();

        // Test multiplication: z = x * y
        // z = [2*1, 3*2, 4*3] = [2, 6, 12]
        let z_mul = x.mul(&y);
        assert_eq!(z_mul.data, vec![2.0, 6.0, 12.0]);
        z_mul.backward();
        // dz/dx = y = [1, 2, 3]
        assert_eq!(*x.grad.borrow(), vec![1.0, 2.0, 3.0]);
        // dz/dy = x = [2, 3, 4]
        assert_eq!(*y.grad.borrow(), vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_mul_scalar() {
        // Test scalar multiplication: z = x * c
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let scalar = 2.5;

        let z = x.mul_scalar(scalar);
        assert_eq!(z.data, vec![5.0, 7.5, 10.0]);

        z.backward();
        // dz/dx = scalar = 2.5 for all elements
        assert_eq!(*x.grad.borrow(), vec![2.5, 2.5, 2.5]);
    }

    #[test]
    fn test_self_binary_ops() {
        // Test operations where both operands are the same tensor
        let mut x = Tensor::new(vec![2.0, 3.0], vec![2]).requires_grad(true);

        // z = x + x = 2x
        let z = x.add(&x);
        assert_eq!(z.data, vec![4.0, 6.0]);
        z.backward();
        // dz/dx = 2 (since x appears twice)
        assert_eq!(*x.grad.borrow(), vec![2.0, 2.0]);

        x.zero_grad();

        // z = x * x = x^2
        let z_sq = x.mul(&x);
        assert_eq!(z_sq.data, vec![4.0, 9.0]);
        z_sq.backward();
        // dz/dx = 2x = [4, 6]
        assert_eq!(*x.grad.borrow(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_complex_expression() {
        // Test a more complex expression: z = (x + y) * (x - y)
        // This is equivalent to: z = x^2 - y^2
        let x = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![2.0], vec![1]).requires_grad(true);

        let sum = x.add(&y); // sum = 5
        let diff = x.sub(&y); // diff = 1
        let z = sum.mul(&diff); // z = 5 * 1 = 5

        assert_eq!(z.data, vec![5.0]);

        z.backward();

        // dz/dx = d/dx(x^2 - y^2) = 2x = 6
        // Manual computation:
        // z = (x+y)(x-y) = x^2 - y^2
        // dz/dx = 2x
        assert_eq!(*x.grad.borrow(), vec![6.0]);

        // dz/dy = d/dy(x^2 - y^2) = -2y = -4
        assert_eq!(*y.grad.borrow(), vec![-4.0]);
    }

    #[test]
    fn test_matmul_forward_and_gradients() {
        // Test matrix multiplication: C = A @ B
        // A: [2x3], B: [3x2] -> C: [2x2]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).requires_grad(true);

        let c = a.matmul(&b);

        // Forward pass verification
        // C[0,0] = 1*1 + 2*3 + 3*5 = 1 + 6 + 15 = 22
        // C[0,1] = 1*2 + 2*4 + 3*6 = 2 + 8 + 18 = 28
        // C[1,0] = 4*1 + 5*3 + 6*5 = 4 + 15 + 30 = 49
        // C[1,1] = 4*2 + 5*4 + 6*6 = 8 + 20 + 36 = 64
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
        assert_eq!(c.shape, vec![2, 2]);

        // Backward pass
        c.backward();

        // dL/dA = grad_output @ B^T
        // grad_output = [[1, 1], [1, 1]]
        // B^T = [[1, 3, 5], [2, 4, 6]]
        // dL/dA = [[1*1 + 1*2, 1*3 + 1*4, 1*5 + 1*6],
        //          [1*1 + 1*2, 1*3 + 1*4, 1*5 + 1*6]]
        //       = [[3, 7, 11], [3, 7, 11]]
        let a_grad = a.grad.borrow();
        assert_eq!(*a_grad, vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);

        // dL/dB = A^T @ grad_output
        // A^T = [[1, 4], [2, 5], [3, 6]]
        // grad_output = [[1, 1], [1, 1]]
        // dL/dB = [[1*1 + 4*1, 1*1 + 4*1],
        //          [2*1 + 5*1, 2*1 + 5*1],
        //          [3*1 + 6*1, 3*1 + 6*1]]
        //       = [[5, 5], [7, 7], [9, 9]]
        let b_grad = b.grad.borrow();
        assert_eq!(*b_grad, vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn test_matmul_chain() {
        // Test chained matrix multiplication: D = (A @ B) @ C
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let b = Tensor::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2]).requires_grad(true);
        let c = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]).requires_grad(true);

        // First matmul: temp = A @ B
        // [[1, 2], [3, 4]] @ [[2, 0], [0, 2]] = [[2, 4], [6, 8]]
        let temp = a.matmul(&b);
        assert_eq!(temp.data, vec![2.0, 4.0, 6.0, 8.0]);

        // Second matmul: d = temp @ C
        // [[2, 4], [6, 8]] @ [[1, 1], [1, 1]] = [[6, 6], [14, 14]]
        let d = temp.matmul(&c);
        assert_eq!(d.data, vec![6.0, 6.0, 14.0, 14.0]);

        d.backward();

        // Verify gradients exist (exact values would require detailed calculation)
        let a_grad = a.grad.borrow();
        let b_grad = b.grad.borrow();
        let c_grad = c.grad.borrow();

        // All gradients should be non-zero
        assert!(a_grad.iter().all(|&g| g != 0.0));
        assert!(b_grad.iter().all(|&g| g != 0.0));
        assert!(c_grad.iter().all(|&g| g != 0.0));
    }

    #[test]
    #[should_panic(expected = "Self-matmul (x.matmul(&x)) is not yet supported")]
    fn test_matmul_self_panic() {
        // Test that self-matmul panics with appropriate message
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let _result = x.matmul(&x);
    }

    #[test]
    fn test_sigmoid_forward_and_backward() {
        // Test sigmoid: σ(x) = 1 / (1 + e^-x)
        // Gradient: dσ/dx = σ(x) * (1 - σ(x))
        let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]).requires_grad(true);
        let y = x.sigmoid();

        // Forward pass verification
        // σ(0) = 1/(1 + 1) = 0.5
        // σ(1) = 1/(1 + e^-1) ≈ 0.7311
        // σ(-1) = 1/(1 + e^1) ≈ 0.2689
        // σ(2) = 1/(1 + e^-2) ≈ 0.8808
        assert!((y.data[0] - 0.5).abs() < 1e-4);
        assert!((y.data[1] - 0.7311).abs() < 1e-4);
        assert!((y.data[2] - 0.2689).abs() < 1e-4);
        assert!((y.data[3] - 0.8808).abs() < 1e-4);

        // Backward pass
        y.backward();

        let x_grad = x.grad.borrow();
        // Gradient: σ(x) * (1 - σ(x))
        // At x=0: 0.5 * 0.5 = 0.25
        // At x=1: 0.7311 * 0.2689 ≈ 0.1966
        // At x=-1: 0.2689 * 0.7311 ≈ 0.1966
        // At x=2: 0.8808 * 0.1192 ≈ 0.1050
        assert!((x_grad[0] - 0.25).abs() < 1e-4);
        assert!((x_grad[1] - 0.1966).abs() < 1e-3);
        assert!((x_grad[2] - 0.1966).abs() < 1e-3);
        assert!((x_grad[3] - 0.1050).abs() < 1e-3);
    }

    #[test]
    fn test_reshape_forward_and_backward() {
        // Test reshape: no computation, just view change
        // Gradient should flow through unchanged
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Reshape from [2, 3] to [3, 2]
        let y = x.reshape(vec![3, 2]);
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]); // Same data

        // Chain with another operation to make gradient non-uniform
        let z = y.mul_scalar(2.0); // z = 2y

        z.backward();

        // Gradient through reshape and mul_scalar: dz/dx = 2.0 for all elements
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_transpose_forward_and_backward() {
        // Test transpose: swap dimensions
        // Gradient should be transposed back
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Transpose from [2, 3] to [3, 2]
        // Original:  [[1, 2, 3],
        //             [4, 5, 6]]
        // Transposed: [[1, 4],
        //              [2, 5],
        //              [3, 6]]
        let y = x.transpose();
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        y.backward();

        // Gradient of transpose is transpose, so all gradients should be 1.0
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_transpose_chain() {
        // Test transpose in a chain: z = (x^T)^T should equal x
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);

        let y = x.transpose(); // [2, 2] -> [2, 2]
        let z = y.transpose(); // [2, 2] -> [2, 2]

        // Double transpose should give back original
        assert_eq!(z.data, x.data);
        assert_eq!(z.shape, x.shape);

        z.backward();

        // Gradient should flow through both transposes
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_reshape_with_computation() {
        // Test reshape combined with actual computation
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);
        let y = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4]).requires_grad(true);

        // Reshape and then add
        let x_reshaped = x.reshape(vec![2, 2]);
        let y_reshaped = y.reshape(vec![2, 2]);
        let z = x_reshaped.add(&y_reshaped);

        assert_eq!(z.data, vec![3.0, 5.0, 7.0, 9.0]);

        z.backward();

        // Gradients should be all 1s (from add)
        let x_grad = x.grad.borrow();
        let y_grad = y.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(*y_grad, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis() {
        // Test sum along axis 0 (rows)
        // [[1, 2, 3],
        //  [4, 5, 6]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Sum along axis 0: [5, 7, 9]
        let y = x.sum_axis(Some(0), false);
        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![5.0, 7.0, 9.0]);

        y.backward();

        // Gradient should be all 1s (each element contributes to one output)
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis_keepdims() {
        // Test sum along axis with keepdims
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Sum along axis 0 with keepdims: [[5, 7, 9]]
        let y = x.sum_axis(Some(0), true);
        assert_eq!(y.shape, vec![1, 3]);
        assert_eq!(y.data, vec![5.0, 7.0, 9.0]);

        y.backward();

        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis_1() {
        // Test sum along axis 1 (columns)
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Sum along axis 1: [6, 15]
        let y = x.sum_axis(Some(1), false);
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![6.0, 15.0]);

        y.backward();

        // Gradient should be all 1s
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        drop(x_grad);
    }

    #[test]
    fn test_sum_axis_1_keepdims() {
        // Test sum along axis 1 with keepdims
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // With keepdims
        let y = x.sum_axis(Some(1), true);
        assert_eq!(y.shape, vec![2, 1]);
        assert_eq!(y.data, vec![6.0, 15.0]);

        y.backward();

        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_mean_axis() {
        // Test mean along axis 0
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Mean along axis 0: [2.5, 3.5, 4.5]
        let y = x.mean_axis(Some(0), false);
        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![2.5, 3.5, 4.5]);

        y.backward();

        // Gradient should be 0.5 for each element (1 / axis_size)
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_mean_axis_keepdims() {
        // Test mean along axis with keepdims
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Mean along axis 0 with keepdims: [[2.5, 3.5, 4.5]]
        let y = x.mean_axis(Some(0), true);
        assert_eq!(y.shape, vec![1, 3]);
        assert_eq!(y.data, vec![2.5, 3.5, 4.5]);

        y.backward();

        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_mean_axis_1() {
        // Test mean along axis 1
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);

        // Mean along axis 1: [2.0, 5.0]
        let y = x.mean_axis(Some(1), false);
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![2.0, 5.0]);

        y.backward();

        // Gradient should be 1/3 for each element
        let x_grad = x.grad.borrow();
        let expected = vec![1.0 / 3.0; 6];
        for (a, b) in x_grad.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sum_mean_chain() {
        // Test chaining sum/mean with other operations
        // [[1, 2],
        //  [3, 4]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);

        // Sum along axis 0: [1+3, 2+4] = [4, 6]
        let y = x.sum_axis(Some(0), false);
        assert_eq!(y.data, vec![4.0, 6.0]);

        // Multiply by scalar: [4*2, 6*2] = [8, 12]
        let z = y.mul_scalar(2.0);
        assert_eq!(z.data, vec![8.0, 12.0]);

        z.backward();

        // Gradient: 2.0 for all elements (from mul_scalar and sum)
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_3d_sum_axis() {
        // Test sum on 3D tensor
        // Shape: [2, 2, 3] = 12 elements
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )
        .requires_grad(true);

        // Sum along axis 2 (last axis)
        let y = x.sum_axis(Some(2), false);
        assert_eq!(y.shape, vec![2, 2]);
        // [1+2+3, 4+5+6, 7+8+9, 10+11+12] = [6, 15, 24, 33]
        assert_eq!(y.data, vec![6.0, 15.0, 24.0, 33.0]);

        y.backward();

        // All gradients should be 1.0
        let x_grad = x.grad.borrow();
        assert_eq!(*x_grad, vec![1.0; 12]);
    }

    #[test]
    fn test_backward_compatibility() {
        // Ensure old sum() and mean() still work
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);

        let s = x.sum();
        assert_eq!(s.data, vec![10.0]);

        let m = x.mean();
        assert_eq!(m.data, vec![2.5]);
    }
}
