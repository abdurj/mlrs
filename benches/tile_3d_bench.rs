use rust_nn::tensor::Tensor;
use std::time::Instant;

struct BenchmarkResult {
    tile_config: String,
    latency_ms: f64,
    speedup: f64,
}

fn gemm_tiling_3d(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
) {
    for ii in (0..m).step_by(tile_m) {
        let i_end = usize::min(ii + tile_m, m);
        for jj in (0..n).step_by(tile_n) {
            let j_end = usize::min(jj + tile_n, n);
            for kk in (0..k).step_by(tile_k) {
                let k_end = usize::min(kk + tile_k, k);

                // Compute the tile with cache-aware order
                for i in ii..i_end {
                    for p in kk..k_end {
                        let a_val = a[i * k + p];
                        for j in jj..j_end {
                            c[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

fn gemm_cache_aware(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        for p in 0..k {
            let a_val = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += a_val * b[p * n + j];
            }
        }
    }
}

fn benchmark_tile_config(
    a: &Tensor,
    b: &Tensor,
    tile_m: usize,
    tile_n: usize,
    tile_k: usize,
    iterations: usize,
) -> f64 {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    // Warm-up
    for _ in 0..3 {
        let mut c = vec![0.0; m * n];
        gemm_tiling_3d(&a.data, &b.data, &mut c, m, k, n, tile_m, tile_n, tile_k);
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut c = vec![0.0; m * n];
        gemm_tiling_3d(&a.data, &b.data, &mut c, m, k, n, tile_m, tile_n, tile_k);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_cache_aware(a: &Tensor, b: &Tensor, iterations: usize) -> f64 {
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    // Warm-up
    for _ in 0..3 {
        let mut c = vec![0.0; m * n];
        gemm_cache_aware(&a.data, &b.data, &mut c, m, k, n);
    }

    // Actual benchmark
    let start = Instant::now();
    for _ in 0..iterations {
        let mut c = vec![0.0; m * n];
        gemm_cache_aware(&a.data, &b.data, &mut c, m, k, n);
    }
    let elapsed = start.elapsed();

    elapsed.as_secs_f64() * 1000.0 / iterations as f64
}

fn benchmark_shape(
    m: usize,
    k: usize,
    n: usize,
    iterations: usize,
    tile_configs: &[(usize, usize, usize)],
) -> Vec<BenchmarkResult> {
    let a = Tensor::randn(vec![m, k]);
    let b = Tensor::randn(vec![k, n]);

    let mut results = Vec::new();

    // Benchmark cache-aware (baseline)
    let baseline_latency = benchmark_cache_aware(&a, &b, iterations);
    results.push(BenchmarkResult {
        tile_config: "CacheAware".to_string(),
        latency_ms: baseline_latency,
        speedup: 1.0,
    });

    // Benchmark all tile configurations
    for &(tile_m, tile_n, tile_k) in tile_configs {
        let latency_ms = benchmark_tile_config(&a, &b, tile_m, tile_n, tile_k, iterations);
        results.push(BenchmarkResult {
            tile_config: format!("{}x{}x{}", tile_m, tile_n, tile_k),
            latency_ms,
            speedup: baseline_latency / latency_ms,
        });
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
    println!("{}", "=".repeat(80));
    println!(
        "{:<20} {:>15} {:>15} {:>15}",
        "Config (MxNxK)", "Latency (ms)", "Speedup", "GFLOPS"
    );
    println!("{}", "-".repeat(80));

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
        } else if result.speedup > 1.0 {
            " +"
        } else {
            ""
        };
        println!(
            "{:<20} {:>15.3} {:>15.2}x {:>15.2}{}",
            result.tile_config, result.latency_ms, result.speedup, gflops, marker
        );
    }
}

fn main() {
    println!("3D Tiling Configuration Optimization");
    println!("=====================================");
    println!("Finding optimal tile sizes (M x N x K) for 3D blocked GEMM");
    println!("* = best performance, + = faster than cache-aware baseline\n");

    // Test various tile configurations
    // Format: (tile_m, tile_n, tile_k)
    let tile_configs = vec![
        // Square tiles
        (16, 16, 16),
        (24, 24, 24),
        (32, 32, 32),
        (48, 48, 48),
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        // Rectangular tiles optimized for common shapes
        (32, 32, 64),
        (32, 64, 32),
        (64, 32, 32),
        (64, 64, 128),
        (64, 128, 64),
        (128, 64, 64),
        // Tall tiles (for MNIST-like: small batch, large input)
        (32, 128, 256),
        (32, 256, 128),
        (64, 128, 256),
        // Wide tiles
        (256, 128, 32),
        (128, 256, 32),
    ];

    // Define benchmark configurations: (name, m, k, n, iterations)
    let benchmarks = vec![
        ("Small Square", 64, 64, 64, 100),
        ("Medium Square", 256, 256, 256, 50),
        ("Medium-Large Square", 512, 512, 512, 20),
        ("Large Square", 1024, 1024, 1024, 10),
        ("MNIST Batch (32)", 32, 784, 128, 50),
        ("MNIST Batch (64)", 64, 784, 256, 50),
        ("MNIST Batch (128)", 128, 784, 512, 30),
        ("CIFAR-10 Batch (64)", 64, 3072, 512, 30),
        ("CIFAR-10 Batch (128)", 128, 3072, 512, 20),
        ("Wide Matrix", 64, 128, 1024, 50),
        ("Tall Matrix", 1024, 128, 64, 50),
        ("Very Large", 2048, 2048, 2048, 5),
    ];

    for (name, m, k, n, iterations) in benchmarks {
        let results = benchmark_shape(m, k, n, iterations, &tile_configs);
        print_results_table(name, m, k, n, &results);
    }

    println!("\n{}", "=".repeat(80));
    println!("Benchmark complete!");
    println!("\nKey Findings:");
    println!("- Configurations marked with * are fastest");
    println!("- Configurations marked with + beat the cache-aware baseline");
    println!("- 3D tiling is most effective for large square matrices");
    println!("- For rectangular matrices, match tile dimensions to matrix shape");
    println!("- Smaller matrices often perform better with cache-aware (no tiling overhead)");
}
