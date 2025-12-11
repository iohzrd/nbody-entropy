//! Benchmark: GPU SVGD vs SGD
//!
//! Compares convergence speed and quality between:
//! - GPU SVGD with 1000 particles
//! - Standard SGD with single particle
//!
//! Run with: cargo run --release --features gpu --bin benchmark

use nbody_entropy::gpu_svgd::GpuSvgd;
use std::time::Instant;

// Training data (same as shader)
const TRAIN_X: [f32; 10] = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
const TRAIN_Y: [f32; 10] = [-1.93, -1.79, -1.52, -0.93, 0.0, 0.93, 1.52, 1.79, 1.93, 1.97];

/// Neural net: y = w2 * tanh(w1 * x)
fn nn_forward(w1: f32, w2: f32, x: f32) -> f32 {
    w2 * (w1 * x).tanh()
}

/// MSE loss
fn nn_loss(w1: f32, w2: f32) -> f32 {
    let mut sum = 0.0;
    for i in 0..10 {
        let pred = nn_forward(w1, w2, TRAIN_X[i]);
        let err = pred - TRAIN_Y[i];
        sum += err * err;
    }
    sum / 10.0
}

/// Gradient of loss
fn nn_gradient(w1: f32, w2: f32) -> [f32; 2] {
    let mut dw1 = 0.0;
    let mut dw2 = 0.0;
    for i in 0..10 {
        let x = TRAIN_X[i];
        let z = w1 * x;
        let a = z.tanh();
        let pred = w2 * a;
        let err = pred - TRAIN_Y[i];
        let tanh_deriv = 1.0 - a * a;
        dw1 += err * w2 * tanh_deriv * x;
        dw2 += err * a;
    }
    [2.0 * dw1 / 10.0, 2.0 * dw2 / 10.0]
}

/// Simple SGD optimizer
struct SGD {
    w1: f32,
    w2: f32,
    lr: f32,
}

impl SGD {
    fn new(w1: f32, w2: f32, lr: f32) -> Self {
        Self { w1, w2, lr }
    }

    fn step(&mut self) {
        let grad = nn_gradient(self.w1, self.w2);
        self.w1 -= self.lr * grad[0];
        self.w2 -= self.lr * grad[1];
    }

    fn loss(&self) -> f32 {
        nn_loss(self.w1, self.w2)
    }
}

fn main() {
    println!("=== GPU SVGD vs SGD Benchmark ===\n");

    // Benchmark parameters
    let num_steps = 1000;
    let particle_counts = [100, 500, 1000];
    let sgd_runs = 100; // Multiple SGD runs from different starting points

    // SGD benchmark (multiple random starts)
    println!("--- SGD (100 random starts, {} steps each) ---", num_steps);
    let mut rng_state = 12345u64;
    let mut sgd_losses = Vec::new();
    let mut sgd_min_losses = Vec::new();

    let sgd_start = Instant::now();
    for _ in 0..sgd_runs {
        // Random init in [-4, 4]
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let w1 = -4.0 + (rng_state & 0xFFFF) as f32 / 65535.0 * 8.0;
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        let w2 = -4.0 + (rng_state & 0xFFFF) as f32 / 65535.0 * 8.0;

        let mut sgd = SGD::new(w1, w2, 0.5);
        let mut min_loss = sgd.loss();

        for _ in 0..num_steps {
            sgd.step();
            min_loss = min_loss.min(sgd.loss());
        }

        sgd_losses.push(sgd.loss());
        sgd_min_losses.push(min_loss);
    }
    let sgd_time = sgd_start.elapsed();

    let sgd_mean: f32 = sgd_losses.iter().sum::<f32>() / sgd_runs as f32;
    let sgd_min: f32 = sgd_losses.iter().cloned().fold(f32::MAX, f32::min);
    let sgd_best_min: f32 = sgd_min_losses.iter().cloned().fold(f32::MAX, f32::min);

    // Count how many found good solutions (loss < 0.1)
    let sgd_success = sgd_losses.iter().filter(|&&l| l < 0.1).count();

    println!("  Time: {:?}", sgd_time);
    println!("  Final loss - Mean: {:.6}, Min: {:.6}", sgd_mean, sgd_min);
    println!("  Best min loss seen: {:.6}", sgd_best_min);
    println!("  Success rate (loss < 0.1): {}/{} ({:.1}%)",
             sgd_success, sgd_runs, 100.0 * sgd_success as f32 / sgd_runs as f32);
    println!();

    // GPU SVGD benchmarks
    for &particle_count in &particle_counts {
        println!("--- GPU SVGD ({} particles, {} steps) ---", particle_count, num_steps);

        let svgd_start = Instant::now();
        let mut svgd = GpuSvgd::new(
            particle_count,
            2,    // 2D (tiny neural net)
            0.5,  // gamma
            0.05, // temperature
            0.2,  // repulsion
            0.3,  // kernel bandwidth
            0.01, // dt
        );

        // Warm up GPU
        for _ in 0..10 {
            svgd.step();
        }
        let init_time = svgd_start.elapsed();

        // Run steps
        let step_start = Instant::now();
        for _ in 0..num_steps {
            svgd.step();
        }
        let step_time = step_start.elapsed();

        // Read results
        let read_start = Instant::now();
        let particles = svgd.read_particles();
        let read_time = read_start.elapsed();

        // Compute statistics
        let losses: Vec<f32> = particles.iter().map(|p| p.energy).collect();
        let mean_loss: f32 = losses.iter().sum::<f32>() / particle_count as f32;
        let min_loss: f32 = losses.iter().cloned().fold(f32::MAX, f32::min);
        let max_loss: f32 = losses.iter().cloned().fold(f32::MIN, f32::max);

        // Count particles in low-loss region
        let low_loss_count = losses.iter().filter(|&&l| l < 0.1).count();

        let steps_per_sec = num_steps as f64 / step_time.as_secs_f64();
        let particles_per_sec = (particle_count * num_steps) as f64 / step_time.as_secs_f64();

        println!("  Init time: {:?}", init_time);
        println!("  Step time: {:?} ({:.0} steps/sec)", step_time, steps_per_sec);
        println!("  Read time: {:?}", read_time);
        println!("  Throughput: {:.0} particle-steps/sec", particles_per_sec);
        println!("  Loss - Mean: {:.6}, Min: {:.6}, Max: {:.6}", mean_loss, min_loss, max_loss);
        println!("  Particles with loss < 0.1: {}/{} ({:.1}%)",
                 low_loss_count, particle_count, 100.0 * low_loss_count as f32 / particle_count as f32);
        println!();
    }

    // Convergence comparison
    println!("=== Convergence Over Time ===\n");
    println!("Tracking min loss at various step counts:\n");

    let checkpoints = [10, 50, 100, 200, 500, 1000];

    // SGD convergence (best of 100 runs)
    print!("Steps:    ");
    for &c in &checkpoints {
        print!("{:>8}", c);
    }
    println!();

    print!("SGD best: ");
    for &checkpoint in &checkpoints {
        let mut best_loss = f32::MAX;
        let mut rng = 42u64;

        for _ in 0..100 {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let w1 = -4.0 + (rng & 0xFFFF) as f32 / 65535.0 * 8.0;
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            let w2 = -4.0 + (rng & 0xFFFF) as f32 / 65535.0 * 8.0;

            let mut sgd = SGD::new(w1, w2, 0.5);
            for _ in 0..checkpoint {
                sgd.step();
            }
            best_loss = best_loss.min(sgd.loss());
        }
        print!("{:>8.5}", best_loss);
    }
    println!();

    // SVGD convergence (1000 particles)
    print!("SVGD min: ");
    let mut svgd = GpuSvgd::new(1000, 2, 0.5, 0.05, 0.2, 0.3, 0.01);
    let mut step_count = 0;
    for &checkpoint in &checkpoints {
        while step_count < checkpoint {
            svgd.step();
            step_count += 1;
        }
        let particles = svgd.read_particles();
        let min_loss: f32 = particles.iter().map(|p| p.energy).fold(f32::MAX, f32::min);
        print!("{:>8.5}", min_loss);
    }
    println!();

    println!("\n=== Summary ===");
    println!("SVGD advantages:");
    println!("  - Explores multiple modes simultaneously");
    println!("  - All particles converge to low-loss regions");
    println!("  - GPU parallelism makes O(n^2) tractable");
    println!("\nSGD advantages:");
    println!("  - Lower overhead for single solution");
    println!("  - No GPU required");
}
