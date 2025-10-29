use rust_nn::{SimpleNet, SGD, Tensor};

/// Generate dummy training data for testing
fn generate_dummy_data(n_samples: usize) -> (Vec<Tensor>, Vec<usize>) {
    // TODO: Create n_samples random images (784 dimensions each)
    // TODO: Create n_samples random labels (0-9)
    // Hint: Use rand::thread_rng() and gen_range()
    todo!("Implement generate_dummy_data")
}

/// Train for one epoch
fn train_epoch(
    model: &mut SimpleNet,
    optimizer: &SGD,
    data: &[Tensor],
    labels: &[usize],
    batch_size: usize,
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
    let batch_size = 32;

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
mod tests {
    // #[test]
    // fn test_matmul() {
    //     // TODO: Test matrix multiplication
    //     // Example: [[1,2],[3,4]] @ [[2,0],[1,2]] = [[4,4],[10,8]]
    // }

    // #[test]
    // fn test_relu() {
    //     // TODO: Test ReLU activation
    //     // Example: relu([-2,-1,0,1,2]) = [0,0,0,1,2]
    // }
}
