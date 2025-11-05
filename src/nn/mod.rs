use crate::Tensor;

/// Optional trait for generic layer interface
pub trait Layer {
    fn forward(&self, input: &Tensor) -> Tensor;
    fn parameters(&mut self) -> Vec<&mut Tensor>;
}

pub trait Loss {
    fn forward(&self, input: &Tensor, targets: &Tensor) -> Tensor;
}

pub mod linear;
pub mod loss;
pub use linear::*;
pub use loss::*;
