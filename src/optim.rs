use crate::tensor::Tensor;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    /// Create a new SGD optimizer with given learning rate
    pub fn new(learning_rate: f32) -> Self {
        // TODO: Create and return SGD with learning_rate
        todo!("Implement SGD::new")
    }

    /// Update parameters using their gradients
    /// Formula: param = param - learning_rate * grad
    pub fn step(&self, parameters: &mut [&mut Tensor]) {
        // TODO: For each parameter:
        //   1. Get its gradient
        //   2. Update: param.data[i] -= learning_rate * grad[i]
        // Hint: Zip parameter data and gradient iterators
        todo!("Implement SGD::step")
    }

    /// Zero out all gradients
    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        // TODO: Call zero_grad() on each parameter
        todo!("Implement SGD::zero_grad")
    }
}
