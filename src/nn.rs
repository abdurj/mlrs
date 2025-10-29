use crate::tensor::Tensor;

/// Linear (fully-connected) layer: y = xW + b
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// Create a new linear layer with Xavier initialization
    pub fn new(in_features: usize, out_features: usize) -> Self {
        // TODO: Xavier init: scale = sqrt(2.0 / in_features)
        // TODO: Create weight tensor of shape [in_features, out_features]
        // TODO: Multiply by scale
        // TODO: Create bias tensor of shape [1, out_features] with zeros
        // TODO: Set both to requires_grad(true)
        todo!("Implement Linear::new")
    }

    /// Forward pass: y = xW + b
    pub fn forward(&self, input: &Tensor) -> Tensor {
        // TODO: Matrix multiply input @ weight
        // TODO: Add bias (broadcast across batch dimension)
        // Hint: For batch_size x out_features output, add bias to each row
        todo!("Implement Linear::forward")
    }

    /// Get mutable references to parameters
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        // TODO: Return vec containing weight and bias
        todo!("Implement Linear::parameters")
    }
}

/// Simple 3-layer neural network for MNIST
pub struct SimpleNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl SimpleNet {
    /// Create a new network with architecture: 784 -> 128 -> 64 -> 10
    pub fn new() -> Self {
        // TODO: Create three linear layers
        // fc1: 784 -> 128 (for 28x28 images)
        // fc2: 128 -> 64
        // fc3: 64 -> 10 (10 classes)
        todo!("Implement SimpleNet::new")
    }

    /// Forward pass through the network
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Pass through fc1, apply ReLU
        // TODO: Pass through fc2, apply ReLU
        // TODO: Pass through fc3 (no activation on output)
        todo!("Implement SimpleNet::forward")
    }

    /// Get all parameters from all layers
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        // TODO: Collect parameters from all three layers
        // Hint: Use extend to combine vectors
        todo!("Implement SimpleNet::parameters")
    }
}

/// Mean Squared Error loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> f32 {
    // TODO: Compute (predictions - targets)^2
    // TODO: Return mean
    todo!("Implement mse_loss")
}

/// Cross-entropy loss for classification
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    // TODO: For each sample, compute:
    //   1. max_logit = max(logits for that sample)
    //   2. log_sum_exp = max_logit + log(sum(exp(logits - max_logit)))
    //   3. loss = log_sum_exp - logit[target_class]
    // TODO: Average loss across all samples
    // Hint: Use log-sum-exp trick for numerical stability
    todo!("Implement cross_entropy_loss")
}

/// Optional trait for generic layer interface
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.parameters()
    }
}
