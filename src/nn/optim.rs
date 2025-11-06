use crate::tensor::Tensor;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    pub learning_rate: f32,
}

impl SGD {
    /// Create a new SGD optimizer with given learning rate
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }

    /// Update parameters using their gradients
    /// Formula: param = param - learning_rate * grad
    pub fn step(&self, parameters: &mut [&mut Tensor]) {
        parameters.iter_mut().for_each(|param| {
            param
                .data
                .iter_mut()
                .zip(param.grad.borrow().iter())
                .for_each(|(p, g)| {
                    *p -= self.learning_rate * *g;
                });
        });
    }

    /// Zero out all gradients
    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        parameters.iter_mut().for_each(|param| {
            param.zero_grad();
        });
    }
}
