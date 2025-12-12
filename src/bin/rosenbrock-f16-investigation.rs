//! Rosenbrock F16 Investigation
//!
//! Why does F16 find better solutions on Rosenbrock?
//! Hypotheses:
//! 1. Implicit regularization - noise from reduced precision escapes local minima
//! 2. Different gradient behavior - f16 gradients have different numerical properties
//! 3. Random seed artifact - just luck with this particular run

use temper::thermodynamic::{LossFunction, ThermodynamicSystem};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║               ROSENBROCK F16 INVESTIGATION                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝\n");

    // Test 1: Multiple runs to check consistency
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 1: CONSISTENCY CHECK (10 runs each)");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_consistency();
    println!();

    // Test 2: Dimension scaling
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 2: DIMENSION SCALING");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    test_dimension_scaling();
    println!();

    // Test 3: Final position analysis
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("TEST 3: FINAL POSITION ANALYSIS");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    analyze_positions();
    println!();

    println!("╔══════════════════════════════════════════════════════════════════════════╗");
    println!("║                     INVESTIGATION COMPLETE                               ║");
    println!("╚══════════════════════════════════════════════════════════════════════════╝");
}

fn run_optimization(use_f16: bool, dim: usize, steps: usize) -> (f32, Vec<f32>) {
    let particle_count = 2000;

    let mut system = if use_f16 {
        ThermodynamicSystem::with_f16_compute(particle_count, dim, 2.0, LossFunction::Rosenbrock)
    } else {
        ThermodynamicSystem::with_loss_function(particle_count, dim, 2.0, LossFunction::Rosenbrock)
    };
    system.set_repulsion_samples(16);

    for step in 0..steps {
        let progress = step as f32 / steps as f32;
        let temp = 2.0 * (0.001_f32 / 2.0).powf(progress);
        system.set_temperature(temp);
        system.step();
    }

    let particles = system.read_particles();
    let best = particles
        .iter()
        .filter(|p| !p.energy.is_nan() && p.energy.is_finite())
        .min_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
        .unwrap();

    let best_pos: Vec<f32> = (0..dim).map(|d| best.pos[d].to_f32()).collect();
    (best.energy, best_pos)
}

fn test_consistency() {
    let dim = 32;
    let steps = 1000;
    let runs = 10;

    let mut f32_results = Vec::new();
    let mut f16_results = Vec::new();

    println!(
        "  Running {} trials for each mode (dim={}, steps={})...",
        runs, dim, steps
    );
    println!();

    for i in 0..runs {
        let (f32_loss, _) = run_optimization(false, dim, steps);
        let (f16_loss, _) = run_optimization(true, dim, steps);
        f32_results.push(f32_loss);
        f16_results.push(f16_loss);
        print!(
            "  Trial {}: F32={:.1}, F16={:.1}",
            i + 1,
            f32_loss,
            f16_loss
        );
        if f16_loss < f32_loss * 0.9 {
            println!(" ← F16 better");
        } else if f32_loss < f16_loss * 0.9 {
            println!(" ← F32 better");
        } else {
            println!();
        }
    }

    // Statistics
    let f32_mean = f32_results.iter().sum::<f32>() / runs as f32;
    let f16_mean = f16_results.iter().sum::<f32>() / runs as f32;
    let f32_min = f32_results.iter().cloned().fold(f32::MAX, f32::min);
    let f16_min = f16_results.iter().cloned().fold(f32::MAX, f32::min);
    let f32_std = (f32_results
        .iter()
        .map(|x| (x - f32_mean).powi(2))
        .sum::<f32>()
        / runs as f32)
        .sqrt();
    let f16_std = (f16_results
        .iter()
        .map(|x| (x - f16_mean).powi(2))
        .sum::<f32>()
        / runs as f32)
        .sqrt();

    let f16_wins = f32_results
        .iter()
        .zip(&f16_results)
        .filter(|(f32_val, f16_val)| **f16_val < **f32_val * 0.9)
        .count();

    println!();
    println!("  Statistics:");
    println!(
        "    F32: mean={:.1}, std={:.1}, min={:.1}",
        f32_mean, f32_std, f32_min
    );
    println!(
        "    F16: mean={:.1}, std={:.1}, min={:.1}",
        f16_mean, f16_std, f16_min
    );
    println!("    F16 wins (>10% better): {}/{}", f16_wins, runs);

    if f16_wins > runs / 2 {
        println!("  → F16 consistently outperforms F32 on Rosenbrock!");
    } else if f16_mean < f32_mean {
        println!("  → F16 slightly better on average, but high variance");
    } else {
        println!("  → Initial result may have been an artifact");
    }
}

fn test_dimension_scaling() {
    let dims = [4, 8, 16, 32, 64];
    let steps = 1000;

    println!(
        "  {:>6} {:>12} {:>12} {:>10}",
        "Dim", "F32 Loss", "F16 Loss", "Winner"
    );
    println!("  {}", "-".repeat(45));

    for &dim in &dims {
        let (f32_loss, _) = run_optimization(false, dim, steps);
        let (f16_loss, _) = run_optimization(true, dim, steps);

        let winner = if f16_loss < f32_loss * 0.9 {
            "F16"
        } else if f32_loss < f16_loss * 0.9 {
            "F32"
        } else {
            "~same"
        };

        println!(
            "  {:>6} {:>12.2} {:>12.2} {:>10}",
            dim, f32_loss, f16_loss, winner
        );
    }

    println!();
    println!("  Analysis:");
    println!("    Rosenbrock has a narrow curved valley.");
    println!("    F16's reduced precision may help escape local minima");
    println!("    or navigate the valley more effectively via implicit noise.");
}

fn analyze_positions() {
    let dim = 8;
    let steps = 2000;

    println!("  Analyzing best positions (dim={})...", dim);
    println!("  Rosenbrock global minimum: (1, 1, ..., 1) with loss = 0");
    println!();

    let (f32_loss, f32_pos) = run_optimization(false, dim, steps);
    let (f16_loss, f16_pos) = run_optimization(true, dim, steps);

    // Distance from optimum (1,1,...,1)
    let f32_dist: f32 = f32_pos
        .iter()
        .map(|x| (x - 1.0).powi(2))
        .sum::<f32>()
        .sqrt();
    let f16_dist: f32 = f16_pos
        .iter()
        .map(|x| (x - 1.0).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("  F32 result:");
    println!("    Loss: {:.4}", f32_loss);
    println!("    Position: {:?}", &f32_pos[..dim.min(8)]);
    println!("    Distance from (1,1,...,1): {:.4}", f32_dist);
    println!();
    println!("  F16 result:");
    println!("    Loss: {:.4}", f16_loss);
    println!("    Position: {:?}", &f16_pos[..dim.min(8)]);
    println!("    Distance from (1,1,...,1): {:.4}", f16_dist);
    println!();

    if f16_dist < f32_dist {
        println!("  → F16 found a position closer to the global minimum!");
    } else if f32_dist < f16_dist {
        println!("  → F32 found a position closer to the global minimum");
    } else {
        println!("  → Similar distances to global minimum");
    }
}
