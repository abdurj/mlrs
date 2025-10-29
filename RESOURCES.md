# Resources & Hints for Rust Neural Network Project

## Quick Reference

### Rust Syntax Refresher

**Vectors:**
```rust
let v = vec![1, 2, 3];
let size = v.len();
let sum: i32 = v.iter().sum();
let mapped: Vec<i32> = v.iter().map(|x| x * 2).collect();
```

**Iterators:**
```rust
// Zip two iterators together
for (a, b) in vec1.iter().zip(vec2.iter()) {
    println!("{} and {}", a, b);
}

// Enumerate with index
for (i, val) in vec.iter().enumerate() {
    println!("Index {}: {}", i, val);
}
```

**Options:**
```rust
if let Some(value) = optional {
    // use value
}

// Or
match optional {
    Some(value) => { /* use value */ },
    None => { /* handle none */ },
}
```

**Random numbers:**
```rust
use rand::Rng;
let mut rng = rand::thread_rng();
let x: f32 = rng.gen_range(-1.0..1.0);
```

---

## Phase 1: Tensor Foundations

### Hint: Tensor::new
```rust
pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
    // Product of shape dimensions
    let expected_size: usize = shape.iter().product();
    assert_eq!(data.len(), expected_size, "Data length must match shape");
    
    Tensor {
        data,
        shape,
        grad: None,
        requires_grad: false,
        grad_fn: None,
    }
}
```

### Hint: Tensor::zeros
```rust
pub fn zeros(shape: Vec<usize>) -> Self {
    let size: usize = shape.iter().product();
    Self::new(vec![0.0; size], shape)
}
```

### Hint: Tensor::randn
```rust
pub fn randn(shape: Vec<usize>) -> Self {
    use rand::Rng;
    let size: usize = shape.iter().product();
    let mut rng = rand::thread_rng();
    let data: Vec<f32> = (0..size)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    Self::new(data, shape)
}
```

### Hint: Matrix Multiplication Algorithm

**Index mapping for 2D matrices:**
- For matrix A with shape [m, k], element at row i, col j is: `A[i * k + j]`
- This is called "row-major" order

**Full matmul implementation:**
```rust
pub fn matmul(&self, other: &Tensor) -> Tensor {
    assert_eq!(self.shape.len(), 2, "First tensor must be 2D");
    assert_eq!(other.shape.len(), 2, "Second tensor must be 2D");
    assert_eq!(self.shape[1], other.shape[0], "Inner dimensions must match");

    let m = self.shape[0];  // rows of first matrix
    let k = self.shape[1];  // cols of first / rows of second
    let n = other.shape[1]; // cols of second matrix

    let mut result = vec![0.0; m * n];

    // Triple nested loop: i for rows, j for cols, p for inner sum
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                // self[i,p] * other[p,j]
                sum += self.data[i * k + p] * other.data[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }

    Tensor::new(result, vec![m, n])
}
```

### Hint: Element-wise Addition
```rust
pub fn add(&self, other: &Tensor) -> Tensor {
    assert_eq!(self.shape, other.shape, "Shapes must match");
    
    let data: Vec<f32> = self.data
        .iter()
        .zip(other.data.iter())
        .map(|(a, b)| a + b)
        .collect();
    
    Tensor::new(data, self.shape.clone())
}
```

### Hint: ReLU Activation
```rust
pub fn relu(&self) -> Tensor {
    let data: Vec<f32> = self.data
        .iter()
        .map(|x| x.max(0.0))
        .collect();
    Tensor::new(data, self.shape.clone())
}
```

---

## Phase 2: Automatic Differentiation

### Understanding Gradients

**Key concept:** For operation `z = f(x, y)`, we need:
- `dz/dx` = how much z changes when x changes
- `dz/dy` = how much z changes when y changes

**Chain rule:** If we know `dL/dz` (gradient from next layer), we compute:
- `dL/dx = dL/dz * dz/dx`
- `dL/dy = dL/dz * dz/dy`

### Hint: MatMulBackward

**Mathematical derivation:**
```
Forward: C = A @ B
Backward given dL/dC:
  dL/dA = dL/dC @ B^T
  dL/dB = A^T @ dL/dC
```

**Implementation:**
```rust
impl GradFn for MatMulBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let m = self.input1.shape[0];
        let k = self.input1.shape[1];
        let n = self.input2.shape[1];

        // grad_input1 = grad_output @ input2^T
        let mut grad1 = vec![0.0; m * k];
        for i in 0..m {
            for j in 0..k {
                let mut sum = 0.0;
                for p in 0..n {
                    // grad_output[i,p] * input2[j,p]
                    sum += grad_output[i * n + p] * self.input2.data[j * n + p];
                }
                grad1[i * k + j] = sum;
            }
        }

        // grad_input2 = input1^T @ grad_output
        let mut grad2 = vec![0.0; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..m {
                    // input1[p,i] * grad_output[p,j]
                    sum += self.input1.data[p * k + i] * grad_output[p * n + j];
                }
                grad2[i * n + j] = sum;
            }
        }

        vec![grad1, grad2]
    }
}
```

### Hint: AddBackward
```rust
impl GradFn for AddBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        // Addition distributes gradient equally
        vec![grad_output.to_vec(), grad_output.to_vec()]
    }
}
```

### Hint: ReLUBackward
```rust
impl GradFn for ReLUBackward {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>> {
        let grad: Vec<f32> = self.input.data
            .iter()
            .zip(grad_output.iter())
            .map(|(x, g)| if *x > 0.0 { *g } else { 0.0 })
            .collect();
        vec![grad]
    }
}
```

### Hint: Attaching grad_fn to Operations

**Pattern for updating operations:**
```rust
pub fn matmul(&self, other: &Tensor) -> Tensor {
    // ... compute forward pass ...
    let mut output = Tensor::new(result, vec![m, n]);
    
    // Add gradient tracking
    if self.requires_grad || other.requires_grad {
        output.requires_grad = true;
        output.grad = Some(vec![0.0; m * n]);
        
        let grad_fn = MatMulBackward {
            input1: self.clone(),
            input2: other.clone(),
        };
        output.grad_fn = Some(Rc::new(RefCell::new(grad_fn)));
    }

    output
}
```

---

## Phase 3: Neural Network Layers

### Hint: Linear Layer Initialization
```rust
pub fn new(in_features: usize, out_features: usize) -> Self {
    // Xavier initialization
    let scale = (2.0 / in_features as f32).sqrt();
    
    let weight = Tensor::randn(vec![in_features, out_features])
        .mul_scalar(scale)
        .requires_grad(true);
    
    let bias = Tensor::zeros(vec![1, out_features])
        .requires_grad(true);

    Linear { weight, bias }
}
```

### Hint: Linear Forward Pass with Bias Broadcasting
```rust
pub fn forward(&self, input: &Tensor) -> Tensor {
    // Matrix multiply
    let output = input.matmul(&self.weight);
    
    // Broadcast bias across batch dimension
    let mut result_data = output.data.clone();
    let batch_size = output.shape[0];
    let out_features = output.shape[1];
    
    for i in 0..batch_size {
        for j in 0..out_features {
            result_data[i * out_features + j] += self.bias.data[j];
        }
    }
    
    Tensor::new(result_data, output.shape)
}
```

### Hint: SimpleNet Architecture
```rust
impl SimpleNet {
    pub fn new() -> Self {
        SimpleNet {
            fc1: Linear::new(784, 128),
            fc2: Linear::new(128, 64),
            fc3: Linear::new(64, 10),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let x = self.fc1.forward(x).relu();
        let x = self.fc2.forward(&x).relu();
        self.fc3.forward(&x)
    }
}
```

### Hint: Cross-Entropy Loss (Numerically Stable)
```rust
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> f32 {
    let mut loss = 0.0;
    let n_classes = logits.shape[1];
    let batch_size = targets.len();
    
    for (i, &target) in targets.iter().enumerate() {
        // Find max logit for numerical stability
        let mut max_logit = f32::NEG_INFINITY;
        for j in 0..n_classes {
            max_logit = max_logit.max(logits.data[i * n_classes + j]);
        }
        
        // Compute log-sum-exp
        let mut sum_exp = 0.0;
        for j in 0..n_classes {
            sum_exp += (logits.data[i * n_classes + j] - max_logit).exp();
        }
        
        let log_sum_exp = max_logit + sum_exp.ln();
        loss += log_sum_exp - logits.data[i * n_classes + target];
    }
    
    loss / batch_size as f32
}
```

---

## Phase 4: Training Loop

### Hint: SGD Optimizer
```rust
impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD { learning_rate }
    }

    pub fn step(&self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            if let Some(ref grad) = param.grad {
                for (p, g) in param.data.iter_mut().zip(grad.iter()) {
                    *p -= self.learning_rate * g;
                }
            }
        }
    }

    pub fn zero_grad(&self, parameters: &mut [&mut Tensor]) {
        for param in parameters {
            param.zero_grad();
        }
    }
}
```

### Hint: Generate Dummy Data
```rust
fn generate_dummy_data(n_samples: usize) -> (Vec<Tensor>, Vec<usize>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut data = Vec::new();
    let mut labels = Vec::new();

    for _ in 0..n_samples {
        // 28x28 image flattened to 784
        let img = Tensor::randn(vec![1, 784]);
        data.push(img);
        
        // Random label 0-9
        labels.push(rng.gen_range(0..10));
    }

    (data, labels)
}
```

### Hint: Training Epoch
```rust
fn train_epoch(
    model: &mut SimpleNet,
    optimizer: &SGD,
    data: &[Tensor],
    labels: &[usize],
    batch_size: usize,
) -> f32 {
    let mut total_loss = 0.0;
    let n_batches = data.len() / batch_size;

    for batch_idx in 0..n_batches {
        let start = batch_idx * batch_size;
        let end = start + batch_size;

        let batch_data = &data[start..end];
        let batch_labels = &labels[start..end];

        // Forward pass
        let mut batch_loss = 0.0;
        for (x, &y) in batch_data.iter().zip(batch_labels.iter()) {
            let pred = model.forward(x);
            let loss = nn::cross_entropy_loss(&pred, &[y]);
            batch_loss += loss;
        }
        batch_loss /= batch_size as f32;

        // Get parameters
        let mut params = model.parameters();
        
        // Simplified: generate random gradients for demonstration
        // In full implementation, backward() would populate these
        for param in params.iter_mut() {
            if let Some(ref mut grad) = param.grad {
                for g in grad.iter_mut() {
                    *g = rand::random::<f32>() * 0.1 - 0.05;
                }
            }
        }
        
        // Update parameters
        optimizer.step(&mut params);
        optimizer.zero_grad(&mut params);

        total_loss += batch_loss;
    }

    total_loss / n_batches as f32
}
```

### Hint: Main Training Loop
```rust
fn main() {
    let mut model = SimpleNet::new();
    let optimizer = SGD::new(0.01);

    let (train_data, train_labels) = generate_dummy_data(1000);
    
    let epochs = 10;
    let batch_size = 32;

    for epoch in 0..epochs {
        let loss = train_epoch(
            &mut model,
            &optimizer,
            &train_data,
            &train_labels,
            batch_size,
        );

        println!("Epoch {}/{} - Loss: {:.4}", epoch + 1, epochs, loss);
    }
}
```

---

## Common Errors & Solutions

### Error: "cannot borrow as mutable more than once"
**Problem:** Rust's borrow checker preventing multiple mutable references.

**Solution:** Use `.iter_mut()` or collect parameters into a Vec first:
```rust
let mut params = model.parameters();
optimizer.step(&mut params);
```

### Error: "attempt to subtract with overflow"
**Problem:** Using `usize` subtraction that goes negative.

**Solution:** Check bounds before subtracting or use `saturating_sub()`.

### Error: "index out of bounds"
**Problem:** Wrong index calculation in matrix operations.

**Solution:** Double-check your index formula. For 2D array stored in 1D:
```rust
// For shape [rows, cols], element at (i, j):
index = i * cols + j
```

### Error: NaN or Infinity in training
**Problem:** 
- Learning rate too high
- Numerical instability in loss computation

**Solution:**
- Lower learning rate to 0.001 or 0.0001
- Use log-sum-exp trick in cross-entropy
- Add gradient clipping

### Error: "expected `Tensor`, found `&Tensor`"
**Problem:** Ownership/reference mismatch.

**Solution:**
- If function expects `Tensor`, clone it: `tensor.clone()`
- If function expects `&Tensor`, pass reference: `&tensor`

---

## Debugging Tips

### Print tensor shapes
```rust
println!("Tensor shape: {:?}", tensor.shape);
println!("Data length: {}", tensor.data.len());
```

### Verify matrix multiplication manually
```rust
// For 2x2 matrices, check individual elements
let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
let b = Tensor::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]);
let c = a.matmul(&b);
println!("Result: {:?}", c.data); // Should be [4, 4, 10, 8]
```

### Check gradient shapes
```rust
if let Some(ref grad) = tensor.grad {
    assert_eq!(grad.len(), tensor.data.len(), "Gradient shape mismatch");
}
```

### Use `cargo check` frequently
Faster than `cargo build`, catches most errors.

### Use `cargo clippy`
Suggests improvements and catches common mistakes.

---

## Performance Tips

### Use release mode
```bash
cargo run --release
```
Can be 10-100x faster than debug mode!

### Profile your code
```rust
use std::time::Instant;

let start = Instant::now();
// ... code to time ...
let duration = start.elapsed();
println!("Time: {:?}", duration);
```

---

## Testing Strategy

### Unit Tests
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_simple() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]);
        let c = a.matmul(&b);
        
        assert_eq!(c.data[0], 4.0);
        assert_eq!(c.data[1], 4.0);
        assert_eq!(c.data[2], 10.0);
        assert_eq!(c.data[3], 8.0);
    }

    #[test]
    fn test_relu_zeros_negatives() {
        let t = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let result = t.relu();
        
        assert_eq!(result.data[0], 0.0);
        assert_eq!(result.data[1], 0.0);
        assert_eq!(result.data[2], 0.0);
        assert_eq!(result.data[3], 1.0);
        assert_eq!(result.data[4], 2.0);
    }
}
```

Run tests: `cargo test`

---

## Mathematical Reference

### Matrix Multiplication
```
C[i,j] = Σ(k=0 to K-1) A[i,k] * B[k,j]

For A(m×k) @ B(k×n) = C(m×n)
```

### Backpropagation
```
Forward: y = f(x)
Backward: dx = dy * df/dx

For chain: z = g(f(x))
dz/dx = dz/dg * dg/df * df/dx
```

### Xavier Initialization
```
W ~ Uniform(-sqrt(6/(n_in + n_out)), sqrt(6/(n_in + n_out)))

Or simpler:
W ~ N(0, sqrt(2/n_in))
```

---

## Next Steps After Completion

1. **Add real MNIST data** - Download and parse actual images
2. **Implement full backprop** - Make gradients actually flow
3. **Add more optimizers** - Adam, RMSprop, AdaGrad
4. **Implement convolution** - Conv2D layers for images
5. **Add batch normalization** - Normalize layer activations
6. **Model saving/loading** - Serialize weights to disk
7. **Validation loop** - Test on held-out data
8. **Learning rate scheduling** - Decay LR during training

---

## Useful Links

- **Rust Book:** https://doc.rust-lang.org/book/
- **Rust by Example:** https://doc.rust-lang.org/rust-by-example/
- **CS231n notes:** http://cs231n.github.io/
- **Matrix calculus:** http://cs231n.stanford.edu/vecDerivs.pdf
- **PyTorch autograd tutorial:** https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html

---

## Minimal Working Example

If you get completely stuck, here's the absolute minimum to get *something* running:

```rust
// tensor.rs - bare minimum
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor { data, shape }
    }
    
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        // Assume 2x2 for simplicity
        let result = vec![
            self.data[0] * other.data[0] + self.data[1] * other.data[2],
            self.data[0] * other.data[1] + self.data[1] * other.data[3],
            self.data[2] * other.data[0] + self.data[3] * other.data[2],
            self.data[2] * other.data[1] + self.data[3] * other.data[3],
        ];
        Tensor::new(result, vec![2, 2])
    }
}

// main.rs - bare minimum
mod tensor;
use tensor::Tensor;

fn main() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b = Tensor::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]);
    let c = a.matmul(&b);
    println!("Result: {:?}", c.data);
}
```

This will at least compile and run! Then build up from there.

---

Good luck with your flight project! Remember: getting stuck is part of learning. Use `println!` liberally to debug, and don't be afraid to simplify if needed. You've got this! ✈️
