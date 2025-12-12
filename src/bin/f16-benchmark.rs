//! F16 vs F32 Compute Benchmark
//!
//! Compares performance and optimization quality between:
//! - F32 compute (full precision position updates)
//! - F16 compute (half precision position updates)
//!
//! Tests across different dimensions and loss functions.

use std::time::Instant;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                    F16 vs F32 COMPUTE BENCHMARK                          ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Test 1: Performance comparison across dimensions
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: PERFORMANCE COMPARISON (steps/sec)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_performance();
    println!();

    // Test 2: Optimization quality comparison
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: OPTIMIZATION QUALITY (final loss)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_quality();
    println!();

    // Test 3: Different loss functions
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: LOSS FUNCTION COMPARISON");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    benchmark_loss_functions();
    println!();

    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                       BENCHMARK COMPLETE                                 ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

fn benchmark_performance() {
    let particle_count = 10_000;
    let steps = 200;
    let dimensions = [8, 32, 64, 128, 256];

    println!("  Particles: {}", particle_count);
    println!("  Steps: {}", steps);
    println!("  Repulsion: disabled (pure update performance)");
    println!();
    println!(
        "  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "Dims", "F32 (s/s)", "F16 (s/s)", "Speedup", "Winner"
    );
    println!("  {}", "-".repeat(60));

    for &dim in &dimensions {
        // F32 compute benchmark
        let mut system_f32 = ThermodynamicSystem::with_loss_function(
            particle_count,
            dim,
            0.01,
            LossFunction::Sphere,
        );
        system_f32.set_repulsion_samples(0);
        system_f32.set_f16_compute(false);

        // Warmup
        for _ in 0..20 {
            system_f32.step();
        }

        let start_f32 = Instant::now();
        for _ in 0..steps {
            system_f32.step();
        }
        let elapsed_f32 = start_f32.elapsed();
        let rate_f32 = steps as f64 / elapsed_f32.as_secs_f64();

        // F16 compute benchmark
        let mut system_f16 = ThermodynamicSystem::with_loss_function(
            particle_count,
            dim,
            0.01,
            LossFunction::Sphere,
        );
        system_f16.set_repulsion_samples(0);
        system_f16.set_f16_compute(true);

        // Warmup
        for _ in 0..20 {
            system_f16.step();
        }

        let start_f16 = Instant::now();
        for _ in 0..steps {
            system_f16.step();
        }
        let elapsed_f16 = start_f16.elapsed();
        let rate_f16 = steps as f64 / elapsed_f16.as_secs_f64();

        let speedup = rate_f16 / rate_f32;
        let winner = if speedup > 1.05 {
            "F16"
        } else if speedup < 0.95 {
            "F32"
        } else {
            "~same"
        };

        println!(
            "  {:>6} {:>12.0} {:>12.0} {:>12.2}x {:>12}",
            dim, rate_f32, rate_f16, speedup, winner
        );
    }

    println!();
    println!("  Note: F16 speedup depends on GPU architecture:");
    println!("    - NVIDIA RTX: ~2x faster (Tensor Cores)");
    println!("    - AMD RDNA2+: ~1.5x faster");
    println!("    - Intel Arc: ~2x faster");
    println!("    - Older GPUs: may be same or slower");
}

fn benchmark_quality() {
    let particle_count = 2000;
    let dim = 64;
    let steps = 2000;

    println!("  Particles: {}", particle_count);
    println!("  Dimensions: {}", dim);
    println!("  Steps: {}", steps);
    println!("  Annealing: T = 2.0 → 0.001");
    println!();
    println!(
        "  {:>12} {:>15} {:>15} {:>12}",
        "Mode", "Final Loss", "Best Loss", "Valid %"
    );
    println!("  {}", "-".repeat(60));

    // F32 optimization
    let mut system_f32 =
        ThermodynamicSystem::with_loss_function(particle_count, dim, 2.0, LossFunction::Rastrigin);
    system_f32.set_repulsion_samples(32);
    system_f32.set_f16_compute(false);

    let mut best_f32 = f32::MAX;
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system_f32.set_temperature(temp);
        system_f32.step();

        if step % 500 == 0 {
            let particles = system_f32.read_particles();
            let min = particles
                .iter()
                .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            if min < best_f32 {
                best_f32 = min;
            }
        }
    }

    let particles_f32 = system_f32.read_particles();
    let final_f32 = particles_f32
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);
    let valid_f32 = particles_f32
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .count() as f32
        / particle_count as f32
        * 100.0;

    println!(
        "  {:>12} {:>15.4} {:>15.4} {:>12.1}",
        "F32", final_f32, best_f32, valid_f32
    );

    // F16 optimization
    let mut system_f16 =
        ThermodynamicSystem::with_loss_function(particle_count, dim, 2.0, LossFunction::Rastrigin);
    system_f16.set_repulsion_samples(32);
    system_f16.set_f16_compute(true);

    let mut best_f16 = f32::MAX;
    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system_f16.set_temperature(temp);
        system_f16.step();

        if step % 500 == 0 {
            let particles = system_f16.read_particles();
            let min = particles
                .iter()
                .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
                .map(|p| p.energy)
                .fold(f32::MAX, f32::min);
            if min < best_f16 {
                best_f16 = min;
            }
        }
    }

    let particles_f16 = system_f16.read_particles();
    let final_f16 = particles_f16
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .map(|p| p.energy)
        .fold(f32::MAX, f32::min);
    let valid_f16 = particles_f16
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .count() as f32
        / particle_count as f32
        * 100.0;

    println!(
        "  {:>12} {:>15.4} {:>15.4} {:>12.1}",
        "F16", final_f16, best_f16, valid_f16
    );

    println!();
    let quality_diff = (final_f16 - final_f32).abs() / final_f32.abs().max(0.001) * 100.0;
    println!("  Quality difference: {:.1}%", quality_diff);
    if quality_diff < 10.0 {
        println!("  → F16 produces comparable optimization quality");
    } else if final_f16 < final_f32 * 1.5 {
        println!("  → F16 quality slightly lower but acceptable");
    } else {
        println!("  → F16 quality significantly lower - consider F32 for precision tasks");
    }
}

fn benchmark_loss_functions() {
    let particle_count = 2000;
    let dim = 32;
    let steps = 1000;

    let loss_functions = [
        ("Sphere", LossFunction::Sphere),
        ("Rastrigin", LossFunction::Rastrigin),
        ("Rosenbrock", LossFunction::Rosenbrock),
        ("Ackley", LossFunction::Ackley),
    ];

    println!("  Particles: {}", particle_count);
    println!("  Dimensions: {}", dim);
    println!("  Steps: {}", steps);
    println!();
    println!(
        "  {:>12} {:>12} {:>12} {:>12}",
        "Loss Fn", "F32 Loss", "F16 Loss", "Quality"
    );
    println!("  {}", "-".repeat(55));

    for (name, loss_fn) in &loss_functions {
        // F32
        let mut system_f32 =
            ThermodynamicSystem::with_loss_function(particle_count, dim, 2.0, loss_fn.clone());
        system_f32.set_repulsion_samples(16);
        system_f32.set_f16_compute(false);

        for step in 0..steps {
            let progress = step as f32 / steps as f32;
            let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
            system_f32.set_temperature(temp);
            system_f32.step();
        }

        let particles_f32 = system_f32.read_particles();
        let loss_f32 = particles_f32
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .map(|p| p.energy)
            .fold(f32::MAX, f32::min);

        // F16
        let mut system_f16 =
            ThermodynamicSystem::with_loss_function(particle_count, dim, 2.0, loss_fn.clone());
        system_f16.set_repulsion_samples(16);
        system_f16.set_f16_compute(true);

        for step in 0..steps {
            let progress = step as f32 / steps as f32;
            let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
            system_f16.set_temperature(temp);
            system_f16.step();
        }

        let particles_f16 = system_f16.read_particles();
        let loss_f16 = particles_f16
            .iter()
            .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
            .map(|p| p.energy)
            .fold(f32::MAX, f32::min);

        let quality = if (loss_f16 - loss_f32).abs() < loss_f32.abs() * 0.1 {
            "~same"
        } else if loss_f16 < loss_f32 {
            "F16 better"
        } else {
            "F32 better"
        };

        println!(
            "  {:>12} {:>12.4} {:>12.4} {:>12}",
            name, loss_f32, loss_f16, quality
        );
    }

    println!();
    println!("  Note: F16 may sometimes find better solutions due to:");
    println!("    - Implicit regularization from reduced precision");
    println!("    - Slightly different random behavior");
}
