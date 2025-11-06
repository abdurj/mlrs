use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use rand::distributions::Uniform;
use rand::prelude::Distribution;
use tracing::instrument;

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
    // Name of the operation for tracing
    name: String,
    // Backward function for this operation
    backward_fn: BackwardFn,
    // Parent nodes in the computation graph (Rc since DAG has no cycles)
    prev: Vec<Rc<GraphNode>>,
}

impl Tensor {
    /// Create a new tensor from data and shape
    #[instrument(skip(data), fields(shape = ?shape, numel = data.len()))]
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
    pub fn with_grad(
        mut result: Tensor,
        parents: Vec<&Tensor>,
        name: &str,
        backward_fn: BackwardFn,
    ) -> Tensor {
        result.requires_grad = true;

        // Collect parent graph nodes (use Rc since DAG has no cycles)
        let parent_nodes: Vec<Rc<GraphNode>> = parents
            .iter()
            .filter_map(|p| p.graph_node.as_ref().map(Rc::clone))
            .collect();

        // Create graph node for this result
        let node = Rc::new(GraphNode {
            name: name.to_string(),
            backward_fn,
            prev: parent_nodes,
        });

        result.graph_node = Some(node);
        result
    }

    /// Helper for binary operations with gradient tracking
    /// Takes two tensors, result data, and a closure that computes gradients
    fn binary_op<F>(
        &self,
        other: &Tensor,
        result_data: Vec<f32>,
        op_name: &str,
        grad_fn: F,
    ) -> Tensor
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
                op_name,
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
    #[instrument(fields(shape = ?shape, numel = shape.iter().product::<usize>()))]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![0.0; numel];

        Tensor::new(data, shape)
    }

    /// Create a tensor filled with ones
    #[instrument(fields(shape = ?shape, numel = shape.iter().product::<usize>()))]
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        let data = vec![1.0; numel];

        Tensor::new(data, shape)
    }

    /// Create a tensor with random values between -1 and 1
    #[instrument(fields(shape = ?shape, numel = shape.iter().product::<usize>()))]
    pub fn randn(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product::<usize>();
        let mut data: Vec<f32> = vec![0.0; numel];

        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        for i in data.iter_mut() {
            *i = uniform.sample(&mut rng);
        }

        Tensor::new(data, shape)
    }

    /// Builder method to enable gradient tracking
    #[instrument(skip(self), fields(shape = ?self.shape, req = req))]
    pub fn requires_grad(mut self, req: bool) -> Self {
        self.requires_grad = req;
        self
    }

    /// Zero out all gradients
    #[instrument(skip(self), fields(shape = ?self.shape, numel = self.numel))]
    pub fn zero_grad(&mut self) {
        for g in self.grad.borrow_mut().iter_mut() {
            *g = 0.0;
        }
    }

    /// Reshape tensor to new shape
    #[instrument(skip(self, new_shape), fields(old_shape = ?self.shape, new_shape = ?new_shape))]
    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        let numel = new_shape.iter().product::<usize>();
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
                "reshape_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("ReshapeBackward").entered();
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
    #[instrument(skip(self, other), fields(shape_a = ?self.shape, shape_b = ?other.shape))]
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
                "matmul_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("MatMulBackward").entered();
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
    #[instrument(skip(self, other), fields(shape = ?self.shape))]
    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a + b)
            .collect();

        self.binary_op(
            other,
            data,
            "add_backward",
            |grad_output, self_grad, other_grad| {
                for i in 0..grad_output.len() {
                    let _span: tracing::span::EnteredSpan = tracing::info_span!("AddBackward").entered();
                    self_grad[i] += grad_output[i];
                    other_grad[i] += grad_output[i];
                }
            },
        )
    }

    /// Element-wise subtraction
    #[instrument(skip(self, other), fields(shape = ?self.shape))]
    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "Shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| a - b)
            .collect();

        self.binary_op(
            other,
            data,
            "sub_backward",
            |grad_output, self_grad, other_grad| {
                for i in 0..grad_output.len() {
                    let _span = tracing::info_span!("SubBackward").entered();
                    self_grad[i] += grad_output[i];
                    other_grad[i] -= grad_output[i]; // Note: negative for subtraction
                }
            },
        )
    }

    /// Element-wise multiplication
    #[instrument(skip(self, other), fields(shape = ?self.shape))]
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

        self.binary_op(
            other,
            data,
            "mul_backward",
            move |grad_output, self_grad, other_grad| {
                let _span = tracing::info_span!("EltwiseMulBackward").entered();
                for i in 0..grad_output.len() {
                    self_grad[i] += grad_output[i] * other_data[i]; // d/dx(x*y) = y
                    other_grad[i] += grad_output[i] * self_data[i]; // d/dy(x*y) = x
                }
            },
        )
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
                "mul_scalar_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("ScalarMulBackward").entered();
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

    /// Element-wise power
    ///
    /// Computes self^exponent for each element
    /// Gradient: d/dx(x^n) = n * x^(n-1)
    #[instrument(skip(self), fields(shape = ?self.shape, exponent = exponent))]
    pub fn pow(&self, exponent: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|x| x.powf(exponent)).collect();
        let result = Tensor::new(data, self.shape.clone());

        if self.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let result_grad = Rc::clone(&result.grad);
            let input_data = self.data.clone();

            Self::with_grad(
                result,
                vec![self],
                "pow_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("PowBackward").entered();
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();

                    for i in 0..grad_output.len() {
                        // d/dx(x^n) = n * x^(n-1)
                        self_g[i] += grad_output[i] * exponent * input_data[i].powf(exponent - 1.0);
                    }
                }),
            )
        } else {
            result
        }
    }

    /// Broadcast addition (2D only)
    ///
    /// Adds `other` to `self` with broadcasting rules for 2D tensors:
    /// - If shapes are identical [M, N] + [M, N], performs element-wise addition
    /// - If `other` has shape [1, N] and `self` has shape [M, N], broadcasts `other` across all M rows
    #[instrument(skip(self, other), fields(shape_a = ?self.shape, shape_b = ?other.shape))]
    pub fn broadcast_add(&self, other: &Tensor) -> Tensor {
        assert_eq!(
            self.shape.len(),
            2,
            "broadcast_add only supports 2D tensors"
        );
        assert_eq!(
            other.shape.len(),
            2,
            "broadcast_add only supports 2D tensors"
        );
        assert_eq!(
            self.shape[1], other.shape[1],
            "Column dimensions must match"
        );

        let rows = self.shape[0];
        let cols = self.shape[1];

        // Check if it's a broadcast case: [M, N] + [1, N]
        let is_broadcast = other.shape[0] == 1;

        if !is_broadcast && self.shape[0] != other.shape[0] {
            panic!(
                "Incompatible shapes for broadcast_add: {:?} and {:?}",
                self.shape, other.shape
            );
        }

        let mut data = vec![0.0; self.numel()];

        if is_broadcast {
            // Broadcasting case: [M, N] + [1, N]
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    data[idx] = self.data[idx] + other.data[j];
                }
            }
        } else {
            // Same shape case: [M, N] + [M, N]
            for (i, d) in data.iter_mut().enumerate().take(self.numel()) {
                *d = self.data[i] + other.data[i];
            }
        }

        let result = Tensor::new(data, self.shape.clone());

        if self.requires_grad || other.requires_grad {
            let self_grad = Rc::clone(&self.grad);
            let other_grad = Rc::clone(&other.grad);
            let result_grad = Rc::clone(&result.grad);

            // Simplification: Don't support self-broadcast_add (x.broadcast_add(&x))
            assert!(
                !Rc::ptr_eq(&self_grad, &other_grad),
                "Self-broadcast_add (x.broadcast_add(&x)) is not supported"
            );

            Self::with_grad(
                result,
                vec![self, other],
                "broadcast_add_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("BroadcastAddBackward").entered();
                    let grad_output = result_grad.borrow();
                    let mut self_g = self_grad.borrow_mut();
                    let mut other_g = other_grad.borrow_mut();

                    if is_broadcast {
                        // Broadcast case: [M, N] + [1, N]
                        // Gradient for self: 1:1 mapping
                        for i in 0..grad_output.len() {
                            self_g[i] += grad_output[i];
                        }

                        // Gradient for other: sum across the broadcast dimension (rows)
                        for j in 0..cols {
                            let mut sum = 0.0;
                            for i in 0..rows {
                                sum += grad_output[i * cols + j];
                            }
                            other_g[j] += sum;
                        }
                    } else {
                        // Same shape: gradient flows equally
                        for i in 0..grad_output.len() {
                            self_g[i] += grad_output[i];
                            other_g[i] += grad_output[i];
                        }
                    }
                }),
            )
        } else {
            result
        }
    }

    /// ReLU activation: max(0, x)
    #[instrument(skip(self), fields(shape = ?self.shape, numel = self.numel))]
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
                "relu_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("ReLUBackward").entered();
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
    #[instrument(skip(self), fields(shape = ?self.shape, numel = self.numel))]
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
                "sigmoid_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("SigmoidBackward").entered();
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
    #[instrument(skip(self), fields(shape = ?self.shape))]
    pub fn sum(&self) -> Tensor {
        self.sum_axis(None, false)
    }

    /// Sum along a specific axis
    ///
    /// # Arguments
    /// * `axis` - The axis to sum along (None means sum all elements)
    /// * `keepdims` - If true, keep the reduced dimension as size 1
    #[instrument(skip(self), fields(shape = ?self.shape, axis = ?axis, keepdims = keepdims))]
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
                        "sum_backward",
                        Box::new(move || {
                            let _span = tracing::info_span!("SumBackward").entered();
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
                        "sum_axis_backward",
                        Box::new(move || {
                            let _span = tracing::info_span!("SumAxisBackward").entered();
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
    #[instrument(skip(self), fields(shape = ?self.shape))]
    pub fn mean(&self) -> Tensor {
        self.mean_axis(None, false)
    }

    /// Mean along a specific axis
    ///
    /// # Arguments
    /// * `axis` - The axis to take mean along (None means mean of all elements)
    /// * `keepdims` - If true, keep the reduced dimension as size 1
    #[instrument(skip(self), fields(shape = ?self.shape, axis = ?axis, keepdims = keepdims))]
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
                        "mean_backward",
                        Box::new(move || {
                            let _span = tracing::info_span!("MeanBackward").entered();
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
                        "mean_axis_backward",
                        Box::new(move || {
                            let _span = tracing::info_span!("MeanAxisBackward").entered();
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
    #[instrument(skip(self), fields(shape = ?self.shape))]
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
                "transpose_backward",
                Box::new(move || {
                    let _span = tracing::info_span!("TransposeBackward").entered();
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
    #[instrument(skip(self), fields(shape = ?self.shape))]
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
                // Traverse parent nodes (already Rc since DAG has no cycles)
                for child in &v.prev {
                    build_topo(child, visited, topo);
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

    // ========================================================================
    // Tensor Creation Tests
    // ========================================================================

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(t.data.len(), 4);
        assert_eq!(t.shape, vec![2, 2]);
    }

    #[test]
    fn test_tensor_zeros() {
        let zeros = Tensor::zeros(vec![2, 2]);
        assert_eq!(zeros.data, vec![0.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_tensor_ones() {
        let ones = Tensor::ones(vec![2, 2]);
        assert_eq!(ones.data, vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_tensor_randn() {
        let rand = Tensor::randn(vec![2, 2]);
        assert_eq!(rand.data.len(), 4);
        assert!(rand.data.iter().all(|&x| (-1.0..=1.0).contains(&x)));
    }

    #[test]
    #[should_panic(expected = "Invalid shape")]
    fn test_invalid_tensor_creation() {
        let _t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![3, 2]);
    }

    // ========================================================================
    // Basic Autograd Tests
    // ========================================================================

    #[test]
    fn test_simple_backward() {
        let x = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        let z = x.add(&y);
        assert_eq!(*x.grad.borrow(), vec![0.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0]);
        assert_eq!(*y.grad.borrow(), vec![1.0]);
    }

    #[test]
    fn test_gradient_accumulation() {
        let x = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        let temp = x.add(&y);
        let z = temp.add(&x);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0]); // x appears twice
        assert_eq!(*y.grad.borrow(), vec![1.0]);
    }

    #[test]
    fn test_zero_grad() {
        let mut x = Tensor::new(vec![1.0], vec![1]).requires_grad(true);
        let y = x.add(&x);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0]);
        x.zero_grad();
        assert_eq!(*x.grad.borrow(), vec![0.0]);
    }

    #[test]
    fn test_composed_operations() {
        let x = Tensor::new(vec![-1.0, 2.0], vec![2]).requires_grad(true);
        let y = Tensor::new(vec![0.5, -1.0], vec![2]).requires_grad(true);
        let sum = x.add(&y);
        let z = sum.relu();
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![0.0, 1.0]);
        assert_eq!(*y.grad.borrow(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_complex_expression() {
        // (x + y) * (x - y) = x^2 - y^2
        let x = Tensor::new(vec![3.0], vec![1]).requires_grad(true);
        let y = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let sum = x.add(&y);
        let diff = x.sub(&y);
        let z = sum.mul(&diff);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![6.0]); // 2x
        assert_eq!(*y.grad.borrow(), vec![-4.0]); // -2y
    }

    // ========================================================================
    // Element-wise Operation Tests
    // ========================================================================

    #[test]
    fn test_add_forward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let z = x.add(&y);
        assert_eq!(z.data, vec![3.0, 5.0, 7.0]);
    }

    #[test]
    fn test_add_backward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let z = x.add(&y);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0]);
        assert_eq!(*y.grad.borrow(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_forward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let z = x.sub(&y);
        assert_eq!(z.data, vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sub_backward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let z = x.sub(&y);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0]);
        assert_eq!(*y.grad.borrow(), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_mul_forward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let z = x.mul(&y);
        assert_eq!(z.data, vec![2.0, 6.0, 12.0]);
    }

    #[test]
    fn test_mul_backward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let z = x.mul(&y);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 2.0, 3.0]); // y
        assert_eq!(*y.grad.borrow(), vec![2.0, 3.0, 4.0]); // x
    }

    #[test]
    fn test_self_add() {
        let x = Tensor::new(vec![2.0, 3.0], vec![2]).requires_grad(true);
        let z = x.add(&x);
        assert_eq!(z.data, vec![4.0, 6.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_self_mul() {
        let x = Tensor::new(vec![2.0, 3.0], vec![2]).requires_grad(true);
        let z = x.mul(&x);
        assert_eq!(z.data, vec![4.0, 9.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![4.0, 6.0]); // 2x
    }

    // ========================================================================
    // Scalar Operation Tests
    // ========================================================================

    #[test]
    fn test_mul_scalar_forward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]);
        let z = x.mul_scalar(2.5);
        assert_eq!(z.data, vec![5.0, 7.5, 10.0]);
    }

    #[test]
    fn test_mul_scalar_backward() {
        let x = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let z = x.mul_scalar(2.5);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.5, 2.5, 2.5]);
    }

    // ========================================================================
    // Power Operation Tests
    // ========================================================================

    #[test]
    fn test_pow_square() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let z = x.pow(2.0);
        assert_eq!(z.data, vec![1.0, 4.0, 9.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0, 4.0, 6.0]); // 2x
    }

    #[test]
    fn test_pow_sqrt() {
        let x = Tensor::new(vec![1.0, 4.0, 9.0], vec![3]).requires_grad(true);
        let z = x.pow(0.5);
        assert_eq!(z.data, vec![1.0, 2.0, 3.0]);
        z.backward();
        assert!((x.grad.borrow()[0] - 0.5).abs() < 1e-6);
        assert!((x.grad.borrow()[1] - 0.25).abs() < 1e-6);
        assert!((x.grad.borrow()[2] - 0.166_666_67).abs() < 1e-6);
    }

    #[test]
    fn test_pow_mse_loss() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let target = Tensor::new(vec![1.5, 1.5, 2.5], vec![3]);
        let diff = pred.sub(&target);
        let squared = diff.pow(2.0);
        assert_eq!(squared.data, vec![0.25, 0.25, 0.25]);
        let loss = squared.mean();
        loss.backward();
        assert!(pred.grad.borrow().iter().all(|&g| !g.is_nan()));
    }

    // ========================================================================
    // Activation Function Tests
    // ========================================================================

    #[test]
    fn test_relu_forward() {
        let x = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let z = x.relu();
        assert_eq!(z.data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let x = Tensor::new(vec![-1.0, 0.0, 1.0, 2.0], vec![4]).requires_grad(true);
        let z = x.relu();
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_forward() {
        let x = Tensor::new(vec![0.0, 1.0, -1.0], vec![3]);
        let z = x.sigmoid();
        assert!((z.data[0] - 0.5).abs() < 1e-6);
        assert!((z.data[1] - 0.7310586).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = Tensor::new(vec![0.0, 1.0, -1.0, 2.0], vec![4]).requires_grad(true);
        let y = x.sigmoid();
        y.backward();
        let x_grad = x.grad.borrow();
        assert!((x_grad[0] - 0.25).abs() < 1e-4);
        assert!((x_grad[1] - 0.1966).abs() < 1e-3);
        assert!((x_grad[2] - 0.1966).abs() < 1e-3);
        assert!((x_grad[3] - 0.1050).abs() < 1e-3);
    }

    // ========================================================================
    // Matrix Multiplication Tests
    // ========================================================================

    #[test]
    fn test_matmul_forward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_backward() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).requires_grad(true);
        let c = a.matmul(&b);
        c.backward();
        assert_eq!(*a.grad.borrow(), vec![11.0, 15.0, 11.0, 15.0]);
        assert_eq!(*b.grad.borrow(), vec![4.0, 4.0, 6.0, 6.0]);
    }

    #[test]
    fn test_matmul_rectangular() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]).requires_grad(true);
        let c = a.matmul(&b);
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
        assert_eq!(c.shape, vec![2, 2]);
        c.backward();
        let a_grad = a.grad.borrow();
        let b_grad = b.grad.borrow();
        assert_eq!(*a_grad, vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);
        assert_eq!(*b_grad, vec![5.0, 5.0, 7.0, 7.0, 9.0, 9.0]);
    }

    #[test]
    fn test_matmul_chain() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2]).requires_grad(true);
        let b = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).requires_grad(true);
        let c = Tensor::new(vec![7.0, 8.0], vec![2, 1]).requires_grad(true);
        let ab = a.matmul(&b);
        let result = ab.matmul(&c);
        assert_eq!(result.shape, vec![1, 1]);
        result.backward();
        assert!(a.grad.borrow().iter().all(|&g| !g.is_nan()));
        assert!(b.grad.borrow().iter().all(|&g| !g.is_nan()));
        assert!(c.grad.borrow().iter().all(|&g| !g.is_nan()));
    }

    #[test]
    #[should_panic(expected = "Self-matmul (x.matmul(&x)) is not yet supported")]
    fn test_matmul_self_panic() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let _result = x.matmul(&x);
    }

    // ========================================================================
    // Shape Manipulation Tests
    // ========================================================================

    #[test]
    fn test_reshape_forward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = x.reshape(vec![3, 2]);
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_backward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.reshape(vec![3, 2]);
        let z = y.mul_scalar(2.0);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_reshape_with_operations() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);
        let y = Tensor::new(vec![2.0, 3.0, 4.0, 5.0], vec![4]).requires_grad(true);
        let x_reshaped = x.reshape(vec![2, 2]);
        let y_reshaped = y.reshape(vec![2, 2]);
        let z = x_reshaped.add(&y_reshaped);
        assert_eq!(z.data, vec![3.0, 5.0, 7.0, 9.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(*y.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_transpose_forward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let y = x.transpose();
        assert_eq!(y.shape, vec![3, 2]);
        assert_eq!(y.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_transpose_backward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.transpose();
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_transpose_double() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let y = x.transpose();
        let z = y.transpose();
        assert_eq!(z.data, x.data);
        assert_eq!(z.shape, x.shape);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    // ========================================================================
    // Reduction Operation Tests
    // ========================================================================

    #[test]
    fn test_sum_all() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);
        let s = x.sum();
        assert_eq!(s.data, vec![10.0]);
        s.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis_0() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.sum_axis(Some(0), false);
        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![5.0, 7.0, 9.0]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis_1() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.sum_axis(Some(1), false);
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![6.0, 15.0]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_axis_keepdims() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.sum_axis(Some(0), true);
        assert_eq!(y.shape, vec![1, 3]);
        assert_eq!(y.data, vec![5.0, 7.0, 9.0]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sum_3d() {
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            vec![2, 2, 3],
        )
        .requires_grad(true);
        let y = x.sum_axis(Some(2), false);
        assert_eq!(y.shape, vec![2, 2]);
        assert_eq!(y.data, vec![6.0, 15.0, 24.0, 33.0]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0; 12]);
    }

    #[test]
    fn test_mean_all() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4]).requires_grad(true);
        let m = x.mean();
        assert_eq!(m.data, vec![2.5]);
        m.backward();
        assert_eq!(*x.grad.borrow(), vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_mean_axis_0() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.mean_axis(Some(0), false);
        assert_eq!(y.shape, vec![3]);
        assert_eq!(y.data, vec![2.5, 3.5, 4.5]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_mean_axis_1() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.mean_axis(Some(1), false);
        assert_eq!(y.shape, vec![2]);
        assert_eq!(y.data, vec![2.0, 5.0]);
        y.backward();
        let expected = [1.0 / 3.0; 6];
        for (a, b) in x.grad.borrow().iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mean_axis_keepdims() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let y = x.mean_axis(Some(0), true);
        assert_eq!(y.shape, vec![1, 3]);
        assert_eq!(y.data, vec![2.5, 3.5, 4.5]);
        y.backward();
        assert_eq!(*x.grad.borrow(), vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
    }

    #[test]
    fn test_sum_mean_chain() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let y = x.sum_axis(Some(0), false);
        let z = y.mul_scalar(2.0);
        assert_eq!(z.data, vec![8.0, 12.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![2.0, 2.0, 2.0, 2.0]);
    }

    // ========================================================================
    // Broadcast Addition Tests
    // ========================================================================

    #[test]
    fn test_broadcast_add_same_shape() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let y = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]).requires_grad(true);
        let z = x.broadcast_add(&y);
        assert_eq!(z.data, vec![6.0, 8.0, 10.0, 12.0]);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(*y.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_broadcast_add_forward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]);
        let z = x.broadcast_add(&bias);
        assert_eq!(z.shape, vec![2, 3]);
        assert_eq!(z.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }

    #[test]
    fn test_broadcast_add_backward() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let bias = Tensor::new(vec![10.0, 20.0, 30.0], vec![1, 3]).requires_grad(true);
        let z = x.broadcast_add(&bias);
        z.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(*bias.grad.borrow(), vec![2.0, 2.0, 2.0]); // sum across rows
    }

    #[test]
    fn test_broadcast_add_with_matmul() {
        // y = xW + b (neural network layer)
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let w = Tensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]).requires_grad(true);
        let b = Tensor::new(vec![0.1, 0.2], vec![1, 2]).requires_grad(true);
        let xw = x.matmul(&w);
        let z = xw.broadcast_add(&b);
        z.backward();
        assert!(x.grad.borrow().iter().all(|&g| !g.is_nan()));
        assert!(w.grad.borrow().iter().all(|&g| !g.is_nan()));
        assert_eq!(*b.grad.borrow(), vec![2.0, 2.0]);
    }

    #[test]
    fn test_broadcast_add_chain() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).requires_grad(true);
        let b1 = Tensor::new(vec![1.0, 2.0], vec![1, 2]).requires_grad(true);
        let b2 = Tensor::new(vec![10.0, 20.0], vec![1, 2]).requires_grad(true);
        let z1 = x.broadcast_add(&b1);
        let z2 = z1.broadcast_add(&b2);
        assert_eq!(z2.data, vec![12.0, 24.0, 14.0, 26.0]);
        z2.backward();
        assert_eq!(*x.grad.borrow(), vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(*b1.grad.borrow(), vec![2.0, 2.0]);
        assert_eq!(*b2.grad.borrow(), vec![2.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "Self-broadcast_add (x.broadcast_add(&x)) is not supported")]
    fn test_broadcast_add_self_panic() {
        let x = Tensor::new(vec![1.0, 2.0], vec![1, 2]).requires_grad(true);
        let _z = x.broadcast_add(&x);
    }
}

// ============================================================================
// Operator Overloads
// ============================================================================

use std::ops::{Add, BitXor, Mul, Sub};

/// Implement the + operator for Tensor references
/// Allows: &tensor1 + &tensor2
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(rhs)
    }
}

/// Implement the - operator for Tensor references
/// Allows: &tensor1 - &tensor2
impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(rhs)
    }
}

/// Implement the * operator for Tensor references (element-wise multiplication)
/// Allows: &tensor1 * &tensor2
impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(rhs)
    }
}

/// Implement scalar multiplication: Tensor * f32
/// Allows: &tensor * scalar
impl Mul<f32> for &Tensor {
    type Output = Tensor;

    fn mul(self, scalar: f32) -> Self::Output {
        self.mul_scalar(scalar)
    }
}

/// Implement scalar multiplication: f32 * Tensor
/// Allows: scalar * &tensor
impl Mul<&Tensor> for f32 {
    type Output = Tensor;

    fn mul(self, tensor: &Tensor) -> Self::Output {
        tensor.mul_scalar(self)
    }
}

/// Implement the ^ operator for power (Tensor ^ f32)
/// Allows: &tensor ^ 2.0
impl BitXor<f32> for &Tensor {
    type Output = Tensor;

    fn bitxor(self, exponent: f32) -> Self::Output {
        self.pow(exponent)
    }
}

#[cfg(test)]
mod operator_tests {
    use super::*;

    #[test]
    fn test_all_operators() {
        // Comprehensive test for all tensor operators: +, -, *, scalar *, ^

        // Test basic arithmetic operators
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).requires_grad(true);

        // Addition: &a + &b
        let add_result = &a + &b;
        assert_eq!(add_result.data, vec![5.0, 7.0, 9.0]);

        // Subtraction: &a - &b
        let sub_result = &a - &b;
        assert_eq!(sub_result.data, vec![-3.0, -3.0, -3.0]);

        // Element-wise multiplication: &a * &b
        let mul_result = &a * &b;
        assert_eq!(mul_result.data, vec![4.0, 10.0, 18.0]);

        // Scalar multiplication: tensor * scalar and scalar * tensor
        let scalar_mul1 = &a * 2.0;
        assert_eq!(scalar_mul1.data, vec![2.0, 4.0, 6.0]);
        let scalar_mul2 = 3.0 * &a;
        assert_eq!(scalar_mul2.data, vec![3.0, 6.0, 9.0]);

        // Power operator: &a ^ exponent
        let pow_result = &a ^ 2.0;
        assert_eq!(pow_result.data, vec![1.0, 4.0, 9.0]);

        // Test chained operations: (a + b) * c - a
        let c = Tensor::new(vec![2.0, 3.0, 4.0], vec![3]).requires_grad(true);
        let sum = &a + &b;
        let prod = &sum * &c;
        let chained = &prod - &a;
        assert_eq!(chained.data, vec![9.0, 19.0, 33.0]); // (1+4)*2-1=9, (2+5)*3-2=19, (3+6)*4-3=33

        // Test gradients with complex expression: y = (2*x + 3)^2
        let x = Tensor::new(vec![1.0, 2.0], vec![2]).requires_grad(true);
        let three = Tensor::new(vec![3.0, 3.0], vec![2]);
        let two_x = 2.0 * &x;
        let expr = &two_x + &three;
        let y = &expr ^ 2.0;
        // y = (2*1+3)^2=25, (2*2+3)^2=49
        assert_eq!(y.data, vec![25.0, 49.0]);

        y.backward();
        // dy/dx = 2*(2x+3)*2 = 4*(2x+3) = [4*5, 4*7] = [20, 28]
        assert_eq!(*x.grad.borrow(), vec![20.0, 28.0]);

        // Test MSE loss pattern: ((pred - target)^2).mean()
        let pred = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let target = Tensor::new(vec![1.5, 1.5, 2.5], vec![3]);
        let diff = &pred - &target;
        let squared = &diff ^ 2.0;
        let loss = squared.mean();
        assert_eq!(loss.data, vec![0.25]);

        loss.backward();
        let pred_grad = pred.grad.borrow();
        assert!((pred_grad[0] - (-0.33333334)).abs() < 1e-6);
        assert!((pred_grad[1] - 0.33333334).abs() < 1e-6);
        assert!((pred_grad[2] - 0.33333334).abs() < 1e-6);

        println!("✓ All operators test passed!");
    }
}
