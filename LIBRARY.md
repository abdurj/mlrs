# rust_nn - A Neural Network Library

This project has been refactored into a library! You can now use `rust_nn` as both a library and a binary.

## Project Structure

```
src/
├── lib.rs       # Library entry point (new!)
├── main.rs      # Binary entry point (train example)
├── tensor.rs    # Tensor operations and autograd
├── nn.rs        # Neural network layers
└── optim.rs     # Optimizers
```

## Using as a Library

### In another Rust project

Add to your `Cargo.toml`:

```toml
[dependencies]
rust_nn = { path = "../path/to/rust_nn" }
```

Then use in your code:

```rust
use rust_nn::{Tensor, Linear, SimpleNet, SGD};

fn main() {
    // Create tensors
    let t1 = Tensor::zeros(vec![2, 2]);
    let t2 = Tensor::ones(vec![2, 2]);
    let t3 = t1.add(&t2);
    
    // Create a neural network
    // let model = SimpleNet::new();
    
    // Create an optimizer
    // let optimizer = SGD::new(0.01);
}
```

### Running the binary

The original `main.rs` is now a binary that can be run with:

```bash
cargo run --bin train
```

Or just:

```bash
cargo run
```

### Building the library

```bash
# Build just the library
cargo build --lib

# Build everything (library + binary)
cargo build

# Run tests
cargo test

# Build documentation
cargo doc --open
```

## API Overview

### `Tensor` - Core tensor operations
- Factory methods: `new()`, `zeros()`, `ones()`, `randn()`
- Operations: `matmul()`, `add()`, `sub()`, `mul()`, `mul_scalar()`
- Activations: `relu()`, `sigmoid()`
- Reductions: `sum()`, `mean()`
- Autograd: `requires_grad()`, `backward()`, `zero_grad()`

### `Linear` - Fully connected layer
- `new(in_features, out_features)` - Create layer with Xavier init
- `forward(input)` - Forward pass
- `parameters()` - Get mutable references to weights and biases

### `SimpleNet` - Example 3-layer network
- `new()` - Create network (784 → 128 → 64 → 10)
- `forward(x)` - Forward pass through all layers

### `SGD` - Stochastic Gradient Descent optimizer
- `new(learning_rate)` - Create optimizer
- `step(parameters)` - Update parameters
- `zero_grad(parameters)` - Zero out gradients

## Examples

Check out `src/main.rs` for a complete training example.

## Next Steps

1. Implement the TODO functions
2. Add more examples in an `examples/` directory
3. Add integration tests in `tests/` directory
4. Publish to crates.io (optional)
