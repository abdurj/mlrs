use rand::Rng;
use rust_nn::optim::SGD;
use rust_nn::{CrossEntropyLoss, Linear, Loss, Tensor};

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
        // fc1: 784 -> 128 (for 28x28 images)
        let fc1 = Linear::new(784, 128);
        // fc2: 128 -> 64
        let fc2 = Linear::new(128, 64);
        // fc3: 64 -> 10 (10 classes)
        let fc3 = Linear::new(64, 10);
        Self { fc1, fc2, fc3 }
    }

    /// Forward pass through the network
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let out1 = self.fc1.forward(x).relu();
        let out2 = self.fc2.forward(&out1).relu();
        self.fc3.forward(&out2)
    }

    /// Get all parameters from all layers
    pub fn parameters(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

struct Batch {
    images: Tensor, // [n_samples, 784]
    labels: Tensor, // [n_samples, 1] class indices as floats
}

/// Generate dummy training data for testing
fn generate_dummy_data(n_samples: usize) -> Batch {
    // Create a single batched tensor [n_samples, 784]
    let mut image_data = Vec::with_capacity(n_samples * 784);
    for _ in 0..n_samples {
        // Generate random image data
        for _ in 0..784 {
            image_data.push(rand::thread_rng().gen_range(-1.0..1.0));
        }
    }
    let images = Tensor::new(image_data, vec![n_samples, 784]);

    // Generate random labels (class indices 0-9) as a tensor
    let label_data: Vec<f32> = (0..n_samples)
        .map(|_| rand::thread_rng().gen_range(0..10) as f32)
        .collect();
    let labels = Tensor::new(label_data, vec![n_samples, 1]);

    Batch { images, labels }
}

/// Train for one epoch
fn train_epoch<L: Loss>(
    model: &mut SimpleNet,
    loss_fn: &L,
    optimizer: &mut SGD,
    data: &Tensor,   // [n_samples, 784]
    labels: &Tensor, // [n_samples, 1] class indices as floats
    batch_size: usize,
) -> f32 {
    let n_samples = data.shape[0];
    let mut total_loss = 0.0;
    let n_batches = n_samples.div_ceil(batch_size);

    for batch_idx in 0..n_batches {
        let start = batch_idx * batch_size;
        let end = ((batch_idx + 1) * batch_size).min(n_samples);
        let curr_batch_size = end - start;

        // Extract batch data [curr_batch_size, 784]
        let batch_data: Vec<f32> = data.data[start * 784..end * 784].to_vec();
        let batch_tensor = Tensor::new(batch_data, vec![curr_batch_size, 784]).requires_grad(true);

        // Extract batch labels [curr_batch_size, 1]
        let batch_label_data: Vec<f32> = labels.data[start..end].to_vec();
        let batch_labels = Tensor::new(batch_label_data, vec![curr_batch_size, 1]);

        // Zero gradients
        optimizer.zero_grad(&mut model.parameters());

        // Forward pass: [curr_batch_size, 784] -> [curr_batch_size, 10]
        let predictions = model.forward(&batch_tensor);

        // Compute loss
        let loss = loss_fn.forward(&predictions, &batch_labels);
        total_loss += loss.data[0];

        // Backward pass
        loss.backward();

        // Optimizer step
        optimizer.step(&mut model.parameters());
    }

    total_loss / n_batches as f32
}

fn main() {
    println!("=== Neural Network Training System ===\n");

    let mut model = SimpleNet::new();

    // Create loss function (separate from optimizer, PyTorch style)
    let loss_fn = CrossEntropyLoss::new();

    // Create optimizer (just needs learning rate)
    let mut optimizer = SGD::new(0.01);

    // Generate dummy data
    println!("Generating training data...");
    let train_batch = generate_dummy_data(1000);
    let _test_batch = generate_dummy_data(200);

    // Training loop
    let epochs = 100;
    let batch_size = 64;

    println!("\nTraining for {} epochs...\n", epochs);

    for epoch in 1..=epochs {
        println!("--- Epoch {}/{} ---", epoch, epochs);
        let avg_loss = train_epoch(
            &mut model,
            &loss_fn,
            &mut optimizer,
            &train_batch.images,
            &train_batch.labels,
            batch_size,
        );
        println!("Average Training Loss: {:.4}\n", avg_loss);
    }
}

#[cfg(test)]
mod tests {}
