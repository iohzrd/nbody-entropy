# N-Body Entropy

An experimental entropy generator that extracts randomness from GPU-accelerated N-body gravitational simulation with chaotic dynamics.

## Concept

Traditional PRNGs use mathematical operations (xorshift, LCG, etc.) to produce pseudo-random sequences. This project explores a different approach: using a **chaotic physical simulation** as the entropy source.

The N-body gravitational problem is inherently chaotic - small differences in initial conditions lead to exponentially diverging trajectories. Combined with a slingshot mechanic that adds tangential velocity boosts on close approach, the system maintains continuous chaotic motion without collapsing into stable configurations.

## How It Works

### The Particle System

64 particles exist in a 2D toroidal space with chaotic n-body dynamics:

**Attractor Particles (8)**
- High mass (10.0)
- Full gravitational interaction with all other attractors
- Create emergent chaotic behavior through n-body dynamics
- Slingshot effect on close approach prevents clumping

**Follower Particles (56)**
- Low mass (1.0)
- Influenced by sampled attractors (not each other)
- Add high-dimensional state space

### Physics

Each particle experiences:
- **Gravitational attraction**: Standard inverse-square law between particles
- **Slingshot effect**: Tangential velocity boost on close approach (radius < 0.08), creating orbital dynamics
- **Velocity damping**: Slight damping (0.999) for stability
- **Toroidal wrapping**: Particles wrap around boundaries

The slingshot mechanic is key - without it, particles eventually clump into a single mass (bad for entropy). With it, particles continuously orbit and interact chaotically.

### Entropy Generation

1. Run N-body physics simulation on GPU
2. Hash particle positions/velocities with BLAKE3
3. Extract random bytes from hash output
4. Repeat

## Performance

| Version | Speed | Notes |
|---------|-------|-------|
| CPU | ~0.87 MB/s | Full n-body simulation |
| GPU | **~2 GB/s** | wgpu compute shaders, batched generation |

```bash
# GPU benchmark
cargo run --release --features gpu -- gpu-benchmark
```

## Usage

### Command Line

```bash
# Run built-in NIST test suite
cargo run --release -- test

# GPU test suite (much faster)
cargo run --release --features gpu -- gpu-test

# Output raw bytes for external tools
cargo run --release --features gpu -- gpu-raw 10000000 | ent

# Continuous stream for dieharder
cargo run --release --features gpu -- gpu-stream | dieharder -a -g 200

# Benchmark
cargo run --release --features gpu -- gpu-benchmark
```

### Visualization

Watch the particles dance:

```bash
cargo run --release --features viz --bin nbody-viz
```

- Orange particles: Attractors (8)
- Blue particles: Followers (56)
- Lines show proximity relationships

### As a Library

```rust
use nbody_entropy::NbodyEntropy;
use rand_core::{RngCore, SeedableRng};

// Create from system time
let mut rng = NbodyEntropy::new();

// Or from explicit seed
let mut rng = NbodyEntropy::from_seed([1, 2, 3, 4, 5, 6, 7, 8]);

// Generate random values (implements RngCore)
let value: u64 = rng.next_u64();

// Fill a buffer
let mut buf = [0u8; 32];
rng.fill_bytes(&mut buf);
```

### GPU-Accelerated Version

```rust
#[cfg(feature = "gpu")]
use nbody_entropy::GpuNbodyEntropy;
use rand_core::{RngCore, SeedableRng};

// GPU version - same API, ~2000x faster
let mut rng = GpuNbodyEntropy::new();
let value = rng.next_u64();
```

## Features

```toml
[dependencies]
nbody-entropy = "0.1"

# With GPU acceleration
nbody-entropy = { version = "0.1", features = ["gpu"] }

# With visualization
nbody-entropy = { version = "0.1", features = ["viz"] }
```

## Statistical Testing

Passes dieharder tests at ~2 GB/s throughput:

```bash
cargo run --release --features gpu -- gpu-stream | dieharder -a -g 200
```

## Limitations

This is an **experimental project**:

1. **Not cryptographically proven** - Needs formal analysis
2. **Deterministic** - Same seed produces same sequence
3. **Requires GPU** - For practical speeds

## How It Differs From Other Chaos PRNGs

Most chaos-based PRNGs use simple systems (logistic map, Lorenz attractor). This uses:

- **High dimensionality**: 64 particles × 4 state variables = 256 dimensions
- **N-body interactions**: O(n²) gravitational relationships
- **Slingshot dynamics**: Prevents collapse into attracting fixed points
- **GPU parallelism**: Each particle computed in parallel

## License

MIT
