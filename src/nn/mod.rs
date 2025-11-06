use crate::Tensor;

/// Optional trait for generic layer interface
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
}

/// Trait for loss functions
pub trait Loss {
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

pub mod linear;
pub mod loss;
pub mod optim;

pub use linear::*;
pub use loss::*;
