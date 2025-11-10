use rust_nn::tensor::Tensor;
use std::time::Instant;

struct BenchmarkResult {
    tile_size: usize,
    latency_ms: f64,
    speedup: f64,
}

fn gemm_tiling_1d(a: &Tensor, b: &Tensor, c: &mut Tensor, tile_size: usize) {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    // iterate over k in tiles
    for kk in (0..k).step_by(tile_size) {
        let k_end = usize::min(kk + tile_size, k);

        // standard matmul for the tile
        for i in 0..m {
            for p in kk..k_end {
                for j in 0..n {
                    c.data[i * n + j] += a.data[i * k + p] * b.data[p * n + j];
                }
            }
        }
    }
}

fn benchmark_tile_size(
    a: &Tensor,
    b: &Tensor,
    tile_size: usize,
    iterations: usize,
) -> f64 {
    // Warm-up
    for _ in 0..3 {
        let mut c = Tensor::zeros(vec![a.shape[0], b.shape[1]]);
        gemm_tiling_1d(a, b, &mut c, tile_size);
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut c = Tensor::zeros(vec![a.shape[0], b.shape[1]]);
        gemm_tiling_1d(a, b, &mut c, tile_size);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_shape(
    m: usize,
    k: usize,
    n: usize,
    iterations: usize,
    tile_sizes: &[usize],
) -> Vec<BenchmarkResult> {
    let a = Tensor::randn(vec![m, k]);
    let b = Tensor::randn(vec![k, n]);

    let mut results = Vec::new();

    for &tile_size in tile_sizes {
        let latency_ms = benchmark_tile_size(&a, &b, tile_size, iterations);
        results.push(BenchmarkResult {
            tile_size,
            latency_ms,
            speedup: 0.0, // Will be calculated later
        });
    }

    // Calculate speedup relative to first tile size (baseline)
    let baseline_latency = results[0].latency_ms;
    for result in &mut results {
        result.speedup = baseline_latency / result.latency_ms;
    }

    results
}

fn print_results_table(
    shape_name: &str,
    m: usize,
    k: usize,
    n: usize,
    results: &[BenchmarkResult],
) {
    println!("\n{} ({}x{}x{})", shape_name, m, k, n);
    println!("{}", "=".repeat(70));
    println!(
        "{:<15} {:>15} {:>15} {:>15}",
        "Tile Size", "Latency (ms)", "Speedup", "GFLOPS"
    );
    println!("{}", "-".repeat(70));

    let ops = 2.0 * m as f64 * k as f64 * n as f64;

    // Find the best result
    let best_latency = results
        .iter()
        .map(|r| r.latency_ms)
        .fold(f64::INFINITY, f64::min);

    for result in results {
        let gflops = ops / (result.latency_ms / 1000.0) / 1e9;
        let marker = if (result.latency_ms - best_latency).abs() < 1e-6 {
            " *"
        } else {
            ""
        };
        println!(
            "{:<15} {:>15.3} {:>15.2}x {:>15.2}{}",
            result.tile_size, result.latency_ms, result.speedup, gflops, marker
        );
    }
}

fn main() {
    println!("Tile Size Optimization Benchmark");
    println!("=================================");
    println!("Finding optimal tile size for 1D tiling GEMM");
    println!("(* indicates best performance)\n");

    // Test a range of tile sizes
    let tile_sizes: Vec<usize> = vec![
        8, 16, 24, 32, 48, 64, 80, 96, 128, 160, 192, 224, 256, 384, 512,
    ];

    // Define benchmark configurations: (name, m, k, n, iterations)
    let benchmarks = vec![
        ("Small Square", 64, 64, 64, 100),
        ("Medium Square", 256, 256, 256, 50),
        ("Large Square", 512, 512, 512, 20),
        ("Very Large", 1024, 1024, 1024, 10),
        ("MNIST Batch (32)", 32, 784, 128, 50),
        ("MNIST Batch (64)", 64, 784, 256, 50),
        ("MNIST Batch (128)", 128, 784, 512, 30),
        ("CIFAR-10 Batch", 128, 3072, 512, 20),
        ("Wide Matrix", 64, 128, 1024, 50),
        ("Tall Matrix", 1024, 128, 64, 50),
    ];

    for (name, m, k, n, iterations) in benchmarks {
        let results = benchmark_shape(m, k, n, iterations, &tile_sizes);
        print_results_table(name, m, k, n, &results);
    }
}
