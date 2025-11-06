use clap::Parser;
use rand::Rng;
use rust_nn::optim::SGD;
use rust_nn::{CrossEntropyLoss, Linear, Loss, Tensor};
use tracing::{info_span, instrument};

/// Simple neural network training for MNIST-like data
#[derive(Parser, Debug)]
#[command(name = "train")]
#[command(about = "Train a simple 3-layer neural network", long_about = None)]
struct Args {
    /// Learning rate for SGD optimizer
    #[arg(short, long, default_value_t = 0.01)]
    learning_rate: f32,

    /// Batch size for training
    #[arg(short, long, default_value_t = 32)]
    batch_size: usize,

    /// Number of training epochs
    #[arg(short, long, default_value_t = 10)]
    epochs: usize,

    /// Number of training samples to generate
    #[arg(short, long, default_value_t = 1000)]
    num_samples: usize,
}

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
    #[instrument(skip(self, x), fields(in_shape = ?x.shape, out_features = self.fc3.out_features))]
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
    let _span = info_span!("train_epoch").entered();

    let n_samples = data.shape[0];
    let mut total_loss = 0.0;
    let n_batches = n_samples.div_ceil(batch_size);

    for batch_idx in 0..n_batches {
        let _batch_span = info_span!("batch", batch_idx).entered();

        let start = batch_idx * batch_size;
        let end = ((batch_idx + 1) * batch_size).min(n_samples);
        let curr_batch_size = end - start;

        // Extract batch data [curr_batch_size, 784]
        let batch_data: Vec<f32> = {
            let _span = info_span!("extract_batch_data").entered();
            data.data[start * 784..end * 784].to_vec()
        };
        let batch_tensor = Tensor::new(batch_data, vec![curr_batch_size, 784]).requires_grad(true);

        // Extract batch labels [curr_batch_size, 1]
        let batch_label_data: Vec<f32> = labels.data[start..end].to_vec();
        let batch_labels = Tensor::new(batch_label_data, vec![curr_batch_size, 1]);

        // Zero gradients
        {
            let _span = info_span!("zero_grad").entered();
            optimizer.zero_grad(&mut model.parameters());
        }

        // Forward pass: [curr_batch_size, 784] -> [curr_batch_size, 10]
        let predictions = { model.forward(&batch_tensor) };

        // Compute loss
        let loss = {
            let _span = info_span!("compute_loss").entered();
            loss_fn.forward(&predictions, &batch_labels)
        };
        total_loss += loss.data[0];

        // Backward pass
        {
            let _span = info_span!("backward").entered();
            loss.backward();
        }

        // Optimizer step
        {
            let _span = info_span!("optimizer_step").entered();
            optimizer.step(&mut model.parameters());
        }
    }

    total_loss / n_batches as f32
}

fn main() {
    // Parse command-line arguments
    let args = Args::parse();

    // Initialize tracing with Chrome trace output
    let (chrome_layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
        .file("./trace.json")
        .build();

    use tracing_subscriber::prelude::*;
    tracing_subscriber::registry().with(chrome_layer).init();

    println!("=== Neural Network Training System ===\n");
    println!("Hyperparameters:");
    println!("  Learning Rate: {}", args.learning_rate);
    println!("  Batch Size: {}", args.batch_size);
    println!("  Epochs: {}", args.epochs);
    println!("  Training Samples: {}\n", args.num_samples);

    let mut model = SimpleNet::new();

    // Create loss function (separate from optimizer, PyTorch style)
    let loss_fn = CrossEntropyLoss::new();

    // Create optimizer with specified learning rate
    let mut optimizer = SGD::new(args.learning_rate);

    // Generate dummy data
    println!("Generating training data...");
    let train_batch = generate_dummy_data(args.num_samples);

    println!("\nTraining for {} epochs...\n", args.epochs);

    {
        let _training_span = info_span!("training_loop").entered();

        for epoch in 1..=args.epochs {
            let _epoch_span = info_span!("epoch", epoch).entered();

            println!("--- Epoch {}/{} ---", epoch, args.epochs);
            let avg_loss = train_epoch(
                &mut model,
                &loss_fn,
                &mut optimizer,
                &train_batch.images,
                &train_batch.labels,
                args.batch_size,
            );
            println!("Average Training Loss: {:.4}\n", avg_loss);
        }
    }

    println!("\n✓ Training complete!");
    println!("Trace written to ./trace.json");
    println!("View it at: https://ui.perfetto.dev/");
}

#[cfg(test)]
mod tests {}
