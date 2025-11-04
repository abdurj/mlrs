use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashSet;
use std::ops::{Add, Mul, Sub};
use std::os::unix::thread;
use std::rc::{Rc, Weak};
use std::result;

use rand::distributions::Uniform;
use rand::prelude::Distribution;
use rand::Rng;

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
    fn with_grad(
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
        Tensor {
            data: self.data.clone(),
            shape: new_shape,
            numel: self.numel,
            grad: Rc::clone(&self.grad),
            requires_grad: self.requires_grad,
            graph_node: self.graph_node.clone(),
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
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();
        
        let result = Tensor::new(data, self.shape.clone());
        
        if self.requires_grad || other.requires_grad {
            // Capture grad Rc references - these are cheap to clone (just increments ref count)
            let self_grad = Rc::clone(&self.grad);
            let other_grad = Rc::clone(&other.grad);
            let result_grad = Rc::clone(&result.grad);
            
            // Create backward closure and attach to result
            Self::with_grad(
                result,
                vec![self, other],
                Box::new(move || {
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();
                    let mut other_g = other_grad.borrow_mut();
                    
                    for i in 0..grad_output.len() {
                        self_g[i] += grad_output[i];
                        other_g[i] += grad_output[i];
                    }
                }),
            )
        } else {
            result
        }
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
        let mut result = Tensor::new(data, self.shape.clone());
        // TODO: grad fn
        result
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
        
        Tensor::new(data, vec![cols, rows])
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
                    if let Some(child) = weak_child.upgrade() {
                        build_topo(&child, visited, topo);
                    }
                    // If upgrade fails, the parent was already dropped - skip it
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
        
        println!("✓ Simple add backward test passed!");
    }

    #[test]
    fn test_backward_chain() {
        // Test: z = (x + y) + x
        // This tests that gradients accumulate properly
        // dz/dx = 2 (appears twice in computation)
        // dz/dy = 1
        let x = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        
        let temp = x.add(&y);  // temp = x + y
        let z = temp.add(&x);   // z = temp + x = (x + y) + x
        
        z.backward();
        
        // x appears twice, so gradient should be 2.0
        assert_eq!(*x.grad.borrow(), vec![2.0]);
        // y appears once, so gradient should be 1.0
        assert_eq!(*y.grad.borrow(), vec![1.0]);
        
        println!("✓ Chain backward test passed!");
        println!("  x.grad = {:?} (expected 2.0)", x.grad.borrow());
        println!("  y.grad = {:?} (expected 1.0)", y.grad.borrow());
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
        
        println!("✓ ReLU backward test passed!");
        println!("  Input: [-1.0, 0.0, 1.0, 2.0]");
        println!("  Gradients: {:?}", *x_grad);
    }

    #[test]
    fn test_backward_multiple_operations() {
        // Test: z = relu(x + y)
        // Tests composition of operations
        let x = Tensor::new(vec![-1.0, 2.0], vec![2]).requires_grad(true);
        let y = Tensor::new(vec![0.5, -1.0], vec![2]).requires_grad(true);
        
        let sum = x.add(&y);    // [-0.5, 1.0]
        let z = sum.relu();      // [0.0, 1.0]
        
        z.backward();
        
        let x_grad = x.grad.borrow();
        let y_grad = y.grad.borrow();
        
        // First element: -1.0 + 0.5 = -0.5, relu -> 0, so gradient = 0
        assert_eq!(x_grad[0], 0.0);
        assert_eq!(y_grad[0], 0.0);
        
        // Second element: 2.0 + (-1.0) = 1.0, relu -> 1.0, so gradient = 1
        assert_eq!(x_grad[1], 1.0);
        assert_eq!(y_grad[1], 1.0);
        
        println!("✓ Multiple operations backward test passed!");
        println!("  x + y = [-0.5, 1.0]");
        println!("  relu(x + y) = [0.0, 1.0]");
        println!("  x.grad = {:?}", *x_grad);
        println!("  y.grad = {:?}", *y_grad);
    }

    #[test]
    fn test_gradient_accumulation() {
        // Test that gradients accumulate correctly across multiple backward passes
        let mut x = Tensor::new(vec![1.0], vec![1]).requires_grad(true);
        
        // First operation
        let y1 = x.add(&x);  // y1 = 2x
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
        
        println!("✓ Gradient accumulation test passed!");
    }
}
