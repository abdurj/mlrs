use crate::{tensor::Tensor, Layer};

/// Linear (fully-connected) layer: y = xW + b
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    in_features: usize,
    out_features: usize,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let mut weight = Tensor::randn(vec![in_features, out_features]).requires_grad(true);
        let bias = Tensor::zeros(vec![1, out_features]).requires_grad(true);

        // Xavier init: scale = sqrt(2.0 / in_features)
        let scale = (2.0 / in_features as f32).sqrt();
        weight.data.iter_mut().for_each(|x| *x *= scale);

        Linear {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    /// Forward pass: y = xW + b
    pub fn forward(&self, input: &Tensor) -> Tensor {
        assert!(input.shape[1] == self.in_features);
        let result = input.matmul(&self.weight).broadcast_add(&self.bias);
        assert!(result.shape[1] == self.out_features);
        result
    }

    /// Get mutable references to parameters
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

impl Layer for Linear {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&mut self) -> Vec<&mut Tensor> {
        self.parameters()
    }
}