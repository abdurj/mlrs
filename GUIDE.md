# Rust Neural Network Training System - 5 Hour Flight Project

## Project Overview
Build a complete neural network training system in Rust that can train on MNIST-like datasets. You'll implement the core tensor operations, automatic differentiation, and training loops from scratch - giving you deep insight into both ML systems and Rust's ownership model.

**Time Allocation:**
- Hour 1: Setup + Tensor fundamentals (30%) 
- Hour 2-3: Autograd system (40%)
- Hour 4: Neural network layers (20%)
- Hour 5: Training loop + testing (10%)

## Prerequisites Setup

### Cargo.toml
```toml
[package]
name = "rust_nn"
version = "0.1.0"
edition = "2021"

[dependencies]
rand = "0.8"
```

That's it! We're building everything else from scratch.

---

## Phase 1: Tensor Foundation (Hour 1)

### Step 1.1: Create the Tensor struct (15 min)

**Goal:** Build a tensor data structure that can hold multi-dimensional arrays and track gradients.

**Key components:**
- `data: Vec<f32>` - the actual values
- `shape: Vec<usize>` - dimensions like [batch, features]
- `grad: Option<Vec<f32>>` - gradients for backprop
- `requires_grad: bool` - whether to compute gradients
- `grad_fn: Option<Rc<RefCell<dyn GradFn>>>` - computation graph node

**Methods to implement:**
- `new(data, shape)` - create from data and shape
- `zeros(shape)` - create tensor of zeros
- `ones(shape)` - create tensor of ones
- `randn(shape)` - create tensor with random values [-1, 1]
- `requires_grad(bool)` - builder pattern to enable gradients
- `zero_grad()` - reset gradients to zero
- `reshape(new_shape)` - change shape while preserving data

### Step 1.2: Matrix Operations (25 min)

**Goal:** Implement the core math operations needed for neural networks.

**Critical operation - Matrix Multiplication:**
```
For A (m×k) @ B (k×n) = C (m×n):
C[i,j] = Σ(p=0 to k-1) A[i,p] * B[p,j]
```

**Methods to implement:**
- `matmul(other)` - matrix multiplication (most important!)
- `add(other)` - element-wise addition
- `sub(other)` - element-wise subtraction
- `mul(other)` - element-wise multiplication
- `mul_scalar(scalar)` - multiply all elements by scalar
- `relu()` - max(0, x) activation
- `sigmoid()` - 1/(1 + e^-x) activation
- `sum()` - sum all elements
- `mean()` - average of all elements

**Important:** Shape validation! Check dimensions match before operations.

### Step 1.3: Test Your Tensors (20 min)

Create comprehensive tests:
1. Basic creation and shape checking
2. Matrix multiplication with known results
3. Element-wise operations
4. Activations (ReLU should zero negatives)
5. Random tensor generation

**Expected matmul result:**
```
[[1, 2],   [[2, 0],    [[4, 4],
 [3, 4]] @  [1, 2]]  =  [10, 8]]
```

---

## Phase 2: Automatic Differentiation (Hours 2-3)

This is the heart of any ML framework. We'll implement reverse-mode autodiff.

### Step 2.1: Gradient Function Trait (20 min)

**Goal:** Create structs that know how to compute gradients for each operation.

**The GradFn trait:**
```rust
pub trait GradFn {
    fn backward(&self, grad_output: &[f32]) -> Vec<Vec<f32>>;
}
```

**Gradient functions to implement:**

1. **MatMulBackward:**
   - Stores: input1, input2
   - Given grad_output, computes:
     - grad_input1 = grad_output @ input2.T
     - grad_input2 = input1.T @ grad_output

2. **AddBackward:**
   - Addition distributes gradients equally
   - grad_input1 = grad_output
   - grad_input2 = grad_output

3. **ReLUBackward:**
   - Stores: input
   - Gradient is 0 where input ≤ 0, else passes through
   - grad_input[i] = input[i] > 0 ? grad_output[i] : 0

**Key insight:** Each operation needs to remember its inputs to compute gradients!

### Step 2.2: Update Tensor Operations with Autograd (30 min)

**Goal:** Connect operations to their gradient functions.

For each operation (`matmul`, `add`, `relu`):
1. Compute forward pass (as before)
2. If any input requires gradients:
   - Set output.requires_grad = true
   - Initialize output.grad
   - Store the GradFn in output.grad_fn

**Example pattern:**
```rust
let mut output = /* compute result */;
if self.requires_grad || other.requires_grad {
    output.requires_grad = true;
    output.grad = Some(vec![0.0; output.data.len()]);
    output.grad_fn = Some(Rc::new(RefCell::new(
        OperationBackward { /* store inputs */ }
    )));
}
```

### Step 2.3: Implement backward() (30 min)

**Goal:** Propagate gradients backward through the computation graph.

**Simplified approach for this project:**
- Initialize output gradient as ones
- Call grad_fn.backward() to get input gradients
- In a full implementation, you'd recursively propagate to all inputs

**The full algorithm (for reference):**
1. Topologically sort computation graph
2. Initialize output gradient
3. For each node in reverse topological order:
   - Call its grad_fn.backward()
   - Accumulate gradients to inputs

### Step 2.4: Test Autograd (30 min)

**Test case 1: Simple multiplication**
```
x = [2, 3]^T  (column vector)
W = [[1, 2]]  (row vector)
y = W @ x = [8]

dy/dx = [1, 2]
dy/dW = [2, 3]
```

**Test case 2: Two-layer network**
```
x -> W1 -> ReLU -> W2 -> y
Test that gradients flow back through ReLU correctly
```

---

## Phase 3: Neural Network Layers (Hour 4)

### Step 3.1: Linear Layer (25 min)

**Goal:** Implement fully-connected layer: y = xW + b

**Structure:**
- `weight: Tensor` of shape [in_features, out_features]
- `bias: Tensor` of shape [1, out_features]

**Xavier Initialization:**
```rust
scale = sqrt(2.0 / in_features)
weight = random_values * scale
```

**Forward pass:**
1. Matrix multiply: input @ weight
2. Broadcast add bias to each row

**Broadcasting example:**
```
[[1, 2],     [0.1, 0.2]    [[1.1, 2.2],
 [3, 4]]  +  [0.1, 0.2]  =  [3.1, 4.2]]
```

### Step 3.2: Build a Simple Network (20 min)

**Goal:** Compose layers into a network.

**Architecture:**
```
Input (784) -> Linear(784, 128) -> ReLU 
            -> Linear(128, 64) -> ReLU
            -> Linear(64, 10) -> Output
```

**Methods needed:**
- `new()` - initialize all layers
- `forward(x)` - pass through all layers
- `parameters()` - collect all weights and biases

### Step 3.3: Loss Functions (15 min)

**1. Mean Squared Error (MSE):**
```
MSE = mean((predictions - targets)²)
```

**2. Cross-Entropy Loss:**
```
For classification:
CE = -log(exp(logit[true_class]) / sum(exp(all_logits)))

Use log-sum-exp trick for numerical stability:
max_logit = max(logits)
CE = max_logit + log(sum(exp(logits - max_logit))) - logit[true_class]
```

---

## Phase 4: Training Loop & SGD (Hour 5)

### Step 4.1: SGD Optimizer (15 min)

**Goal:** Implement Stochastic Gradient Descent.

**Update rule:**
```
param_new = param_old - learning_rate * gradient
```

**Methods:**
- `new(learning_rate)` - create optimizer
- `step(parameters)` - update all parameters using their gradients
- `zero_grad(parameters)` - reset all gradients to zero

**Critical:** Always zero gradients before next forward pass!

### Step 4.2: Training Loop (25 min)

**Goal:** Put everything together into a training loop.

**Training epoch structure:**
```
for each batch:
    1. Forward pass: predictions = model(input)
    2. Compute loss: loss = loss_fn(predictions, targets)
    3. Backward pass: loss.backward()
    4. Optimizer step: optimizer.step()
    5. Zero gradients: optimizer.zero_grad()
```

**Batch processing:**
- Split data into batches of size 32
- Process each batch independently
- Average loss across batches

### Step 4.3: Dummy Data Generation (10 min)

**For testing without real data:**
- Generate random 784-dimensional vectors (28×28 flattened)
- Random labels 0-9
- Create train set (~1000 samples) and test set (~200 samples)

### Step 4.4: Full Training Script (10 min)

**Main training loop:**
```
for epoch in 1..10:
    total_loss = 0
    for batch in data:
        loss = train_batch(batch)
        total_loss += loss
    print(f"Epoch {epoch}: Loss = {total_loss / num_batches}")
```

**Expected output:**
- Loss should be computable (even if random without real backprop)
- Training should complete all epochs
- System should not crash or panic

---

## What You've Learned

**Rust Concepts:**
- Ownership and borrowing (managing tensor data)
- Trait objects (`dyn GradFn`)
- `Rc<RefCell<>>` for shared mutable state
- Generic programming with traits
- Module organization
- Builder pattern (`requires_grad`)

**ML Systems Concepts:**
- Tensor operations and broadcasting
- Computational graph construction
- Reverse-mode automatic differentiation
- Layer abstractions
- Optimizer design patterns
- Training loop architecture

---

## Testing Your System

### Checkpoint 1 (Hour 1):
```rust
cargo run
```
Should see tensor operations working.

### Checkpoint 2 (Hour 3):
Autograd structure compiles and gradient functions exist.

### Checkpoint 3 (Hour 5):
Full training loop runs for 10 epochs.

---

## Extensions (If You Have Extra Time)

### 1. Better Backward Pass (20 min)
Implement full topological sorting for proper backpropagation:
- Build computation graph explicitly
- Toposort using Kahn's algorithm
- Propagate gradients in reverse order

### 2. Save/Load Models (15 min)
```rust
impl SimpleNet {
    pub fn save(&self, path: &str) -> std::io::Result<()> {
        // Serialize weights to file
    }
    
    pub fn load(path: &str) -> std::io::Result<Self> {
        // Deserialize weights from file
    }
}
```

### 3. More Optimizers (20 min)
- **Adam:** Adaptive learning rates with momentum
- **RMSprop:** Root mean square propagation
- **Momentum:** Accelerated gradient descent

### 4. Batch Normalization (30 min)
```
y = (x - mean) / sqrt(variance + epsilon) * gamma + beta
```

### 5. Convolutional Layers (45 min)
Implement Conv2D for image processing.

---

## Common Pitfalls & Solutions

**Problem:** Borrow checker errors with gradients
**Solution:** Use `Rc<RefCell<>>` for shared mutable state

**Problem:** Shape mismatches in matmul
**Solution:** Always print shapes when debugging, add assertions

**Problem:** Gradients not flowing back
**Solution:** Ensure grad_fn is stored and backward() is called

**Problem:** NaN or infinity in training
**Solution:** Check learning rate (try 0.01 or lower), use gradient clipping

**Problem:** Slow performance
**Solution:** Use `cargo run --release` for optimized builds

---

## Resources

- Rust Book: https://doc.rust-lang.org/book/
- Autograd tutorial: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html
- Matrix calculus: http://cs231n.stanford.edu/vecDerivs.pdf

---

## Success Criteria

✅ Tensors can be created and manipulated  
✅ Matrix multiplication works correctly  
✅ Gradient functions compile and are attached to operations  
✅ Neural network layers can be created  
✅ Training loop runs without panicking  
✅ Loss values are computed and printed  

**Bonus:**
✅ Real gradients flow backward through the network  
✅ Model actually learns on dummy data  
✅ Can save and load model weights

---

## Final Notes

This project gives you a solid foundation in:
- Rust systems programming
- ML framework internals
- Computational graph design
- Automatic differentiation

The patterns you learn here are used in PyTorch, TensorFlow, JAX, and other ML frameworks. You're building the engine that powers modern deep learning!

Good luck, and enjoy your flight! ✈️
