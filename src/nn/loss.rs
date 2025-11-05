use crate::{tensor::Tensor, Layer};

struct MSELoss {}

impl MSELoss {

    fn forward(predictions: &Tensor, targets: &Tensor) -> Tensor {
        let result = predictions.sub(targets);
        result.mul(&result).mean_axis(None, false)
    }

}
/// Mean Squared Error loss
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    // TODO: Compute (predictions - targets)^2
    // TODO: Return mean
    todo!("Implement mse_loss")
}

/// Cross-entropy loss for classification
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // TODO: For each sample, compute:
    //   1. max_logit = max(logits for that sample)
    //   2. log_sum_exp = max_logit + log(sum(exp(logits - max_logit)))
    //   3. loss = log_sum_exp - logit[target_class]
    // TODO: Average loss across all samples
    // Hint: Use log-sum-exp trick for numerical stability
    todo!("Implement cross_entropy_loss")
}
