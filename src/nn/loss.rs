use crate::tensor::Tensor;
use crate::nn::Loss;
use std::rc::Rc;

/// Mean Squared Error loss
pub struct MSELoss;

impl MSELoss {
    pub fn new() -> Self {
        MSELoss
    }
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for MSELoss {
    /// MSE = mean((predictions - targets)^2)
    fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        (predictions - targets).pow(2.0).mean()
    }
}

/// Mean Squared Error loss function (standalone)
/// MSE = mean((predictions - targets)^2)
pub fn mse_loss(predictions: &Tensor, targets: &Tensor) -> Tensor {
    (predictions - targets).pow(2.0).mean()
}

/// Convert integer class labels to one-hot encoded tensor
/// 
/// targets: [batch_size] - integer class labels (0 to num_classes-1)
/// num_classes: total number of classes
/// 
/// Returns: [batch_size, num_classes] tensor with 1.0 at target indices, 0.0 elsewhere
/// Helper function to convert class indices to one-hot encoded vectors
#[allow(dead_code)]
fn one_hot_encode(targets: &[usize], num_classes: usize) -> Vec<f32> {
    let batch_size = targets.len();
    let mut data = vec![0.0; batch_size * num_classes];
    
    for (i, &target) in targets.iter().enumerate() {
        assert!(target < num_classes, "Target class {} out of bounds (num_classes={})", target, num_classes);
        data[i * num_classes + target] = 1.0;
    }
    
    data
}

/// Compute log-sum-exp for numerical stability: log(sum(exp(x)))
/// Uses the identity: log(sum(exp(x))) = max + log(sum(exp(x - max)))
fn log_sum_exp(values: &[f32]) -> f32 {
    let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = values.iter().map(|&x| (x - max_val).exp()).sum();
    max_val + sum_exp.ln()
}

/// Compute softmax probabilities for a batch of logits
/// Returns a vector of probabilities with the same shape as input
fn compute_softmax(logits: &[f32], batch_size: usize, num_classes: usize) -> Vec<f32> {
    let mut softmax = vec![0.0; batch_size * num_classes];
    
    for i in 0..batch_size {
        let start = i * num_classes;
        let end = start + num_classes;
        let row = &logits[start..end];
        
        // Compute log-sum-exp for this row
        let lse = log_sum_exp(row);
        
        // Compute softmax: exp(x - log_sum_exp)
        for j in 0..num_classes {
            softmax[start + j] = (row[j] - lse).exp();
        }
    }
    
    softmax
}

/// Cross-entropy loss for classification
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss
    }

    /// Compute cross-entropy loss
    /// logits: [batch_size, num_classes] - raw model outputs
    /// targets: [batch_size] - integer class labels (0 to num_classes-1)
    pub fn compute(&self, logits: &Tensor, targets: &[usize]) -> Tensor {
        cross_entropy_loss(logits, targets)
    }
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(&self, logits: &Tensor, targets: &Tensor) -> Tensor {
        let target_indices: Vec<usize> = targets.data.iter().map(|&x| x as usize).collect();
        self.compute(logits, &target_indices)
    }
}

/// Cross-entropy loss for classification with autograd support (standalone function)
/// 
/// logits: [batch_size, num_classes] - raw model outputs
/// targets: [batch_size] - integer class labels (0 to num_classes-1)
/// 
/// Returns: scalar loss tensor with gradients flowing through logits
/// 
/// Formula: CE = -mean(sum(one_hot_targets * log(softmax(logits))))
/// Gradient: d_logits = (softmax(logits) - one_hot_targets) / batch_size
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    assert_eq!(logits.shape.len(), 2, "logits must be 2D [batch, classes]");
    let batch_size = logits.shape[0];
    let num_classes = logits.shape[1];
    assert_eq!(targets.len(), batch_size, "targets length must match batch_size");

    // Compute loss value using log-sum-exp for numerical stability
    let mut total_loss = 0.0;
    for (i, &target_class) in targets.iter().enumerate().take(batch_size) {
        let offset = i * num_classes;
        let sample_logits = &logits.data[offset..offset + num_classes];

        let lse = log_sum_exp(sample_logits);

        total_loss += lse - sample_logits[target_class];
    }

    let loss_value = total_loss / (batch_size as f32);
    let mut result = Tensor::new(vec![loss_value], vec![1]);

    // Set up autograd if logits requires gradients
    if logits.requires_grad {
        result.requires_grad = true;
        
        // Clone data needed for backward pass
        let logits_grad = Rc::clone(&logits.grad);
        let logits_data = logits.data.clone();
        let targets_vec = targets.to_vec();
        let shape = logits.shape.clone();
        
        // Create backward function
        let backward_fn = Box::new(move || {
            let mut grad = logits_grad.borrow_mut();
            let batch_size = shape[0];
            let num_classes = shape[1];
            
            // Compute softmax probabilities for all samples
            let softmax_probs = compute_softmax(&logits_data, batch_size, num_classes);
            
            // Gradient: (softmax - one_hot) / batch_size
            for (i, &target_class) in targets_vec.iter().enumerate().take(batch_size) {
                let offset = i * num_classes;
                
                for j in 0..num_classes {
                    // one_hot is 1.0 at target class, 0.0 elsewhere
                    let one_hot = if j == target_class { 1.0 } else { 0.0 };
                    
                    // d_logits[i,j] = (softmax[i,j] - one_hot[i,j]) / batch_size
                    grad[offset + j] += (softmax_probs[offset + j] - one_hot) / (batch_size as f32);
                }
            }
        });
        
        // Use with_grad to hook into autograd
        result = Tensor::with_grad(result, vec![logits], backward_fn);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse_loss() {
        // Test basic MSE computation
        let predictions = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let targets = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);

        let loss = mse_loss(&predictions, &targets);

        // Perfect predictions should give zero loss
        assert_eq!(loss.data[0], 0.0);

        println!("✓ MSE loss basic test passed!");
    }

    #[test]
    fn test_mse_loss_with_error() {
        // Test MSE with actual errors
        let predictions = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).requires_grad(true);
        let targets = Tensor::new(vec![1.5, 1.5, 2.5], vec![3]);

        let loss = mse_loss(&predictions, &targets);

        // MSE = mean([(-0.5)^2, (0.5)^2, (0.5)^2]) = mean([0.25, 0.25, 0.25]) = 0.25
        assert_eq!(loss.data[0], 0.25);

        println!("✓ MSE loss with error test passed!");
    }

    #[test]
    fn test_mse_loss_gradients() {
        // Test gradient computation for MSE loss
        let predictions = Tensor::new(vec![2.0, 4.0, 6.0], vec![3]).requires_grad(true);
        let targets = Tensor::new(vec![1.0, 3.0, 5.0], vec![3]);

        // Use mse_loss function directly
        let loss = mse_loss(&predictions, &targets);

        // MSE = mean([(1)^2, (1)^2, (1)^2]) = 1.0
        assert_eq!(loss.data[0], 1.0);

        loss.backward();

        // Gradient: d/dpred MSE = 2/n * (pred - target)
        // = 2/3 * [1, 1, 1] = [2/3, 2/3, 2/3]
        let pred_grad = predictions.grad.borrow();
        assert!((pred_grad[0] - 0.6666667).abs() < 1e-6);
        assert!((pred_grad[1] - 0.6666667).abs() < 1e-6);
        assert!((pred_grad[2] - 0.6666667).abs() < 1e-6);

        println!("✓ MSE loss gradients test passed!");
    }

    #[test]
    fn test_mse_loss_2d() {
        // Test MSE with 2D tensors (batch of predictions)
        let predictions =
            Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).requires_grad(true);
        let targets = Tensor::new(vec![1.5, 2.5, 3.5, 3.5, 4.5, 5.5], vec![2, 3]);

        // Use mse_loss function directly
        let loss = mse_loss(&predictions, &targets);

        // MSE = mean([(-0.5)^2, (-0.5)^2, (-0.5)^2, (0.5)^2, (0.5)^2, (0.5)^2])
        // = mean([0.25, 0.25, 0.25, 0.25, 0.25, 0.25]) = 0.25
        assert_eq!(loss.data[0], 0.25);

        loss.backward();

        // Check gradients flow correctly
        let pred_grad = predictions.grad.borrow();
        assert!(pred_grad.iter().all(|&g| !g.is_nan()));

        println!("✓ MSE loss 2D test passed!");
    }

    #[test]
    fn test_mse_loss_large_errors() {
        // Test MSE penalizes large errors quadratically
        let pred_small = Tensor::new(vec![1.0], vec![1]).requires_grad(true);
        let pred_large = Tensor::new(vec![2.0], vec![1]).requires_grad(true);
        let target = Tensor::new(vec![0.0], vec![1]);

        let loss_small = mse_loss(&pred_small, &target);
        let loss_large = mse_loss(&pred_large, &target);

        // Small error: (1-0)^2 = 1
        // Large error: (2-0)^2 = 4
        assert_eq!(loss_small.data[0], 1.0);
        assert_eq!(loss_large.data[0], 4.0);

        // Large error should be 4x worse than small error
        assert_eq!(loss_large.data[0] / loss_small.data[0], 4.0);

        println!("✓ MSE loss large errors test passed!");
    }

    // ========================================================================
    // Cross-Entropy Loss Tests
    // ========================================================================

    #[test]
    fn test_cross_entropy_basic() {
        // Test basic cross-entropy computation
        // Perfect prediction: logits have very high score for correct class
        let logits = Tensor::new(
            vec![
                10.0, 0.0, 0.0,  // Sample 0: predicts class 0
                0.0, 10.0, 0.0,  // Sample 1: predicts class 1
            ],
            vec![2, 3],
        ).requires_grad(true);
        
        let targets = vec![0, 1];
        
        let loss = cross_entropy_loss(&logits, &targets);
        
        // Loss should be very small (close to 0) for perfect predictions
        assert!(loss.data[0] < 0.01, "Loss should be close to 0 for perfect predictions");
        
        println!("✓ Cross-entropy basic test passed!");
    }

    #[test]
    fn test_cross_entropy_wrong_prediction() {
        // Test with completely wrong predictions
        let logits = Tensor::new(
            vec![
                0.0, 10.0, 0.0,  // Sample 0: predicts class 1, but target is 0
                10.0, 0.0, 0.0,  // Sample 1: predicts class 0, but target is 1
            ],
            vec![2, 3],
        ).requires_grad(true);
        
        let targets = vec![0, 1];
        
        let loss = cross_entropy_loss(&logits, &targets);
        
        // Loss should be high for wrong predictions
        assert!(loss.data[0] > 5.0, "Loss should be high for wrong predictions");
        
        println!("✓ Cross-entropy wrong prediction test passed!");
    }

    #[test]
    fn test_cross_entropy_gradients() {
        // Test gradient computation
        let logits = Tensor::new(
            vec![
                1.0, 2.0, 3.0,   // Sample 0: target class 2
                3.0, 1.0, 2.0,   // Sample 1: target class 0
            ],
            vec![2, 3],
        ).requires_grad(true);
        
        let targets = vec![2, 0];
        
        let loss = cross_entropy_loss(&logits, &targets);
        
        // Verify loss is computed
        assert!(loss.data[0] > 0.0);
        
        // Call autograd backward
        loss.backward();
        
        // Check gradients are computed
        let grad = logits.grad.borrow();
        
        // For sample 0, target is class 2, so:
        // - grad[0,2] should be negative (softmax[2] - 1)/batch_size
        // - grad[0,0] and grad[0,1] should be positive (softmax values)/batch_size
        assert!(grad[2] < 0.0, "Gradient at target class should be negative");
        assert!(grad[0] > 0.0, "Gradient at non-target class should be positive");
        assert!(grad[1] > 0.0, "Gradient at non-target class should be positive");
        
        // Gradients should sum to ~0 for each sample
        let grad_sum_sample0 = grad[0] + grad[1] + grad[2];
        let grad_sum_sample1 = grad[3] + grad[4] + grad[5];
        assert!(grad_sum_sample0.abs() < 1e-6, "Gradients should sum to 0 per sample");
        assert!(grad_sum_sample1.abs() < 1e-6, "Gradients should sum to 0 per sample");
        
        println!("✓ Cross-entropy gradients test passed!");
    }

    #[test]
    fn test_cross_entropy_uniform_distribution() {
        // Test with uniform logits (maximum uncertainty)
        let logits = Tensor::new(
            vec![
                1.0, 1.0, 1.0,   // Sample 0: uniform
                2.0, 2.0, 2.0,   // Sample 1: uniform (different scale)
            ],
            vec![2, 3],
        ).requires_grad(true);
        
        let targets = vec![0, 1];
        
        let loss = cross_entropy_loss(&logits, &targets);
        
        // Loss should be -log(1/3) ≈ 1.0986 for uniform distribution over 3 classes
        let expected_loss = -(1.0f32 / 3.0).ln();
        assert!((loss.data[0] - expected_loss).abs() < 0.01);
        
        println!("✓ Cross-entropy uniform distribution test passed!");
    }

    #[test]
    fn test_cross_entropy_numerical_stability() {
        // Test with very large logits (tests numerical stability of log-sum-exp)
        let logits = Tensor::new(
            vec![
                100.0, 200.0, 150.0,   // Very large values
                -100.0, -50.0, -75.0,  // Very small values
            ],
            vec![2, 3],
        ).requires_grad(true);
        
        let targets = vec![1, 1];
        
        let loss = cross_entropy_loss(&logits, &targets);
        
        // Should not produce NaN or Inf
        assert!(!loss.data[0].is_nan(), "Loss should not be NaN");
        assert!(!loss.data[0].is_infinite(), "Loss should not be infinite");
        
        // Call backward to test gradient computation stability
        loss.backward();
        let grad = logits.grad.borrow();
        assert!(grad.iter().all(|&g| !g.is_nan()), "Gradients should not be NaN");
        assert!(grad.iter().all(|&g| !g.is_infinite()), "Gradients should not be infinite");
        
        println!("✓ Cross-entropy numerical stability test passed!");
    }
}
