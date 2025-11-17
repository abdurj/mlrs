use crate::tensor::Tensor;
use tracing::instrument;

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
    #[instrument(skip(self, parameters), fields(num_params = parameters.len(), lr = self.learning_rate))]
    pub fn step(&self, parameters: &mut [&mut Tensor]) {
        parameters.iter_mut().for_each(|param| {
            let grad_storage = param.grad_storage();
            param
                .data_storage_mut()
                .borrow_mut()
                .iter_mut()
                .zip(grad_storage.borrow().iter())
                .for_each(|(p, g)| {
                    *p -= self.learning_rate * *g;
                });
        });
    }

    /// Zero out all gradients
    #[instrument(skip(self, parameters), fields(num_params = parameters.len()))]
    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        parameters.iter_mut().for_each(|param| {
            param.zero_grad();
        });
    }
}
