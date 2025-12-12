//! Performance Profiling Benchmark
//!
//! Identifies bottlenecks in the thermodynamic particle system:
//! - Workgroup size impact
//! - Dimension scaling analysis
//! - Repulsion vs update time breakdown
//! - Memory bandwidth limits

use std::time::Instant;
use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

fn main() {
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                    PERFORMANCE PROFILING                                 ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝\n"
    );

    // Test 1: Detailed dimension scaling analysis
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: DIMENSION SCALING ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_dimension_scaling_detailed();
    println!();

    // Test 2: Repulsion samples impact
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: REPULSION SAMPLES IMPACT");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_repulsion_scaling();
    println!();

    // Test 3: Memory bandwidth analysis
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: MEMORY BANDWIDTH ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_memory_bandwidth();
    println!();

    // Test 4: Loss function complexity impact
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 4: LOSS FUNCTION COMPLEXITY");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_loss_complexity();
    println!();

    // Summary
    println!(
        "{}",
        "╔══════════════════════════════════════════════════════════════════════════╗"
    );
    println!(
        "{}",
        "║                      PROFILING COMPLETE                                  ║"
    );
    println!(
        "{}",
        "╚══════════════════════════════════════════════════════════════════════════╝"
    );
}

fn test_dimension_scaling_detailed() {
    let particle_count = 10_000;
    let steps = 100;

    // Test fine-grained dimension scaling
    let dimensions = [8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256];

    println!("  Particle count: {}", particle_count);
    println!("  Steps: {}", steps);
    println!("  Repulsion samples: 0 (pure optimization - isolates update cost)");
    println!();
    println!(
        "  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "Dims", "Time (ms)", "Steps/sec", "Slowdown", "us/dim/part"
    );
    println!("  {}", "-".repeat(60));

    let mut baseline_rate = 0.0;

    for &dim in &dimensions {
        let mut system = ThermodynamicSystem::with_loss_function(
            particle_count,
            dim,
            0.01, // Low temperature for gradient-dominated updates
            LossFunction::Sphere,
        );
        system.set_repulsion_samples(0); // Disable repulsion to isolate update cost

        // Warmup
        for _ in 0..20 {
            system.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();

        let rate = steps as f64 / elapsed.as_secs_f64();
        if dim == 8 {
            baseline_rate = rate;
        }
        let slowdown = baseline_rate / rate;

        // Microseconds per dimension per particle per step
        let us_per_dim_part = (elapsed.as_micros() as f64)
            / (steps as f64 * dim as f64 * particle_count as f64)
            * 1000.0; // Convert to nanoseconds

        println!(
            "  {:>6} {:>12.1} {:>12.0} {:>12.2}x {:>12.3}",
            dim,
            elapsed.as_secs_f64() * 1000.0,
            rate,
            slowdown,
            us_per_dim_part
        );
    }

    println!();
    println!("  Analysis:");
    println!("    - If slowdown is linear with dims: compute-bound (expected)");
    println!("    - If slowdown is superlinear: memory bandwidth or cache issues");
}

fn test_repulsion_scaling() {
    let particle_count = 5_000;
    let dim = 64;
    let steps = 50;

    let repulsion_samples = [0, 8, 16, 32, 64, 128, 256];

    println!("  Particle count: {}", particle_count);
    println!("  Dimensions: {}", dim);
    println!("  Steps: {}", steps);
    println!();
    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "Samples", "Time (ms)", "Steps/sec", "Cost/sample"
    );
    println!("  {}", "-".repeat(50));

    let mut baseline_time = 0.0;

    for &samples in &repulsion_samples {
        let mut system =
            ThermodynamicSystem::with_loss_function(particle_count, dim, 1.0, LossFunction::Sphere);
        system.set_repulsion_samples(samples);

        // Warmup
        for _ in 0..10 {
            system.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

        if samples == 0 {
            baseline_time = elapsed_ms;
        }

        let rate = steps as f64 / elapsed.as_secs_f64();
        let cost_per_sample = if samples > 0 {
            (elapsed_ms - baseline_time) / (samples as f64)
        } else {
            0.0
        };

        println!(
            "  {:>8} {:>12.1} {:>12.0} {:>12.3}",
            samples, elapsed_ms, rate, cost_per_sample
        );
    }

    println!();
    println!("  Analysis:");
    println!("    - Repulsion cost should be O(K) where K = samples");
    println!("    - High cost/sample indicates inner loop inefficiency");
}

fn test_memory_bandwidth() {
    let dim = 256;
    let steps = 50;

    // Test different particle counts to see memory bandwidth limits
    let particle_counts = [1_000, 5_000, 10_000, 25_000, 50_000, 100_000];

    println!("  Dimensions: {} (max)", dim);
    println!("  Particle struct size: 520 bytes");
    println!("  Repulsion: disabled (isolates memory access pattern)");
    println!();
    println!(
        "  {:>10} {:>12} {:>12} {:>12} {:>12}",
        "Particles", "Buffer MB", "Time (ms)", "Steps/sec", "GB/s"
    );
    println!("  {}", "-".repeat(65));

    for &count in &particle_counts {
        let mut system =
            ThermodynamicSystem::with_loss_function(count, dim, 0.01, LossFunction::Sphere);
        system.set_repulsion_samples(0);

        // Warmup
        for _ in 0..10 {
            system.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();

        let buffer_mb = (count * 520) as f64 / (1024.0 * 1024.0);
        let rate = steps as f64 / elapsed.as_secs_f64();

        // Estimate bandwidth: 2 reads + 1 write per step = 3 * buffer_size
        let bytes_per_step = count as f64 * 520.0 * 3.0;
        let gb_per_sec = (bytes_per_step * rate) / (1024.0 * 1024.0 * 1024.0);

        println!(
            "  {:>10} {:>12.1} {:>12.1} {:>12.0} {:>12.1}",
            count,
            buffer_mb,
            elapsed.as_secs_f64() * 1000.0,
            rate,
            gb_per_sec
        );
    }

    println!();
    println!("  Analysis:");
    println!("    - GB/s plateau indicates memory bandwidth limit");
    println!("    - Typical GPU bandwidth: 200-900 GB/s");
}

fn test_loss_complexity() {
    let particle_count = 10_000;
    let dim = 64;
    let steps = 100;

    println!("  Particle count: {}", particle_count);
    println!("  Dimensions: {}", dim);
    println!("  Repulsion: disabled");
    println!();
    println!(
        "  {:>15} {:>12} {:>12} {:>12}",
        "Loss Function", "Time (ms)", "Steps/sec", "Relative"
    );
    println!("  {}", "-".repeat(55));

    let loss_functions = [
        ("Sphere", LossFunction::Sphere),
        ("Rastrigin", LossFunction::Rastrigin),
        ("Rosenbrock", LossFunction::Rosenbrock),
        ("Ackley", LossFunction::Ackley),
        ("Schwefel", LossFunction::Schwefel),
    ];

    let mut baseline_rate = 0.0;

    for (name, loss_fn) in &loss_functions {
        let mut system =
            ThermodynamicSystem::with_loss_function(particle_count, dim, 0.01, loss_fn.clone());
        system.set_repulsion_samples(0);

        // Warmup
        for _ in 0..20 {
            system.step();
        }

        let start = Instant::now();
        for _ in 0..steps {
            system.step();
        }
        let elapsed = start.elapsed();

        let rate = steps as f64 / elapsed.as_secs_f64();
        if *name == "Sphere" {
            baseline_rate = rate;
        }
        let relative = baseline_rate / rate;

        println!(
            "  {:>15} {:>12.1} {:>12.0} {:>12.2}x",
            name,
            elapsed.as_secs_f64() * 1000.0,
            rate,
            relative
        );
    }

    println!();
    println!("  Analysis:");
    println!("    - Sphere is simplest (baseline)");
    println!("    - Rastrigin/Ackley have trig functions (slower)");
    println!("    - High relative cost = loss function dominates");
}
