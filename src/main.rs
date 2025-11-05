use rust_nn::{Linear, Tensor, SGD};

/// Simple 3-layer neural network for MNIST
#[allow(dead_code)]
pub struct SimpleNet {
    fc1: Linear,
    fc2: Linear,
    fc3: Linear,
}

impl Default for SimpleNet {
    fn default() -> Self {
        Self::new()
    }
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
    pub fn forward(&self, _x: &Tensor) -> Tensor {
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

/// Generate dummy training data for testing
#[allow(dead_code)]
fn generate_dummy_data(_n_samples: usize) -> (Vec<Tensor>, Vec<usize>) {
    // TODO: Create n_samples random images (784 dimensions each)
    // TODO: Create n_samples random labels (0-9)
    // Hint: Use rand::thread_rng() and gen_range()
    todo!("Implement generate_dummy_data")
}

/// Train for one epoch
#[allow(dead_code)]
fn train_epoch(
    _model: &mut SimpleNet,
    _optimizer: &SGD,
    _data: &[Tensor],
    _labels: &[usize],
    _batch_size: usize,
) -> f32 {
    // TODO: Iterate through data in batches
    // For each batch:
    //   1. Forward pass: get predictions
    //   2. Compute loss
    //   3. Backward pass (simplified for now)
    //   4. Optimizer step
    //   5. Zero gradients
    // Return average loss
    todo!("Implement train_epoch")
}

fn main() {
    println!("=== Neural Network Training System ===\n");

    // TODO: Create model
    // let mut model = SimpleNet::new();

    // TODO: Create optimizer
    // let optimizer = SGD::new(0.01);

    // TODO: Generate dummy data
    println!("Generating training data...");
    // let (train_data, train_labels) = generate_dummy_data(1000);
    // let (test_data, test_labels) = generate_dummy_data(200);

    // TODO: Training loop
    let epochs = 10;
    let _batch_size = 32;

    println!("\nTraining for {} epochs...\n", epochs);

    // TODO: For each epoch:
    //   1. Call train_epoch
    //   2. Print epoch number and loss

    println!("\n✓ Training complete!");
    println!("\n=== Project Complete! ===");
    println!("You've built:");
    println!("  • Tensor operations with shape handling");
    println!("  • Automatic differentiation system");
    println!("  • Neural network layers (Linear + ReLU)");
    println!("  • SGD optimizer");
    println!("  • Training loop");
}

#[cfg(test)]
mod tests {}
