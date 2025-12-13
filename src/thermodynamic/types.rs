//! Core types for the thermodynamic particle system

use bytemuck::{Pod, Zeroable};
use half::f16;

/// Maximum dimensions supported (for network weights, images, etc.)
/// 4096 supports 32x32 RGB images (3072 dims) or 64x64 grayscale (4096 dims)
pub const MAX_DIMENSIONS: usize = 4096;

/// Maximum particles supported (GPU memory dependent)
pub const MAX_PARTICLES: usize = 500_000;

/// Operating mode determined by temperature
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermodynamicMode {
    /// T < 0.01: Pure gradient descent
    Optimize,
    /// 0.01 <= T <= 1.0: Bayesian sampling
    Sample,
    /// T > 1.0: Entropy extraction
    Entropy,
}

impl ThermodynamicMode {
    pub fn from_temperature(t: f32) -> Self {
        if t < 0.01 {
            Self::Optimize
        } else if t <= 1.0 {
            Self::Sample
        } else {
            Self::Entropy
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::Optimize => "OPTIMIZE",
            Self::Sample => "SAMPLE",
            Self::Entropy => "ENTROPY",
        }
    }
}

/// Loss function to optimize
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[repr(u32)]
pub enum LossFunction {
    #[default]
    /// Original 2D neural net
    NeuralNet2D = 0,
    /// N-dimensional multimodal
    Multimodal = 1,
    /// Classic banana valley, min at (1,1,...,1)
    Rosenbrock = 2,
    /// Highly multimodal, min at origin
    Rastrigin = 3,
    /// Flat outer region, hole at center
    Ackley = 4,
    /// Simple convex baseline
    Sphere = 5,
    /// Real MLP on XOR problem (9 params)
    MlpXor = 6,
    /// Real MLP on spiral classification
    MlpSpiral = 7,
    /// Deep MLP: 2->4->4->1 (37 params) on circles dataset
    MlpDeep = 8,
    /// Deceptive - global min at (420.97, ...) far from origin
    Schwefel = 9,
    /// Custom expression-based loss function
    Custom = 10,
    /// Product of cosines, min at origin
    Griewank = 11,
    /// Multimodal with sin² terms, min at (1,1,...,1)
    Levy = 12,
    /// Simple polynomial, min at (-2.903534, ..., -2.903534)
    StyblinskiTang = 13,
}

/// Particle state in the thermodynamic system
///
/// Uses f16 for positions to match GPU memory layout directly.
/// Size: 4096*2 + 4 + 4 = 8200 bytes
/// MAX_DIMENSIONS=4096 supports 32x32 RGB images or 64x64 grayscale
///
/// To convert positions to f32 for computation: `p.pos[d].to_f32()`
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ThermodynamicParticle {
    pub pos: [f16; MAX_DIMENSIONS], // 2048 bytes
    pub energy: f32,                // 4 bytes
    pub entropy_bits: u32,          // 4 bytes
}

impl Default for ThermodynamicParticle {
    fn default() -> Self {
        Self {
            pos: [f16::ZERO; MAX_DIMENSIONS],
            energy: 0.0,
            entropy_bits: 0,
        }
    }
}

/// GPU uniforms for the compute shader
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct Uniforms {
    pub particle_count: u32,
    pub dim: u32,
    pub gamma: f32,
    pub temperature: f32,
    pub repulsion_strength: f32,
    pub kernel_bandwidth: f32,
    pub dt: f32,
    pub seed: u32,
    pub mode: u32,
    pub loss_fn: u32,
    pub repulsion_samples: u32,
    pub use_f16_compute: u32,
}

/// Statistics about current particle distribution
#[derive(Debug, Clone)]
pub struct ThermodynamicStats {
    pub mean_energy: f32,
    pub min_energy: f32,
    pub max_energy: f32,
    pub spread: f32,
    pub low_energy_fraction: f32,
    pub mode: ThermodynamicMode,
    pub temperature: f32,
}

/// Population diversity metrics for analyzing particle distribution
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Mean pairwise distance between particles (higher = more diverse)
    pub mean_pairwise_distance: f32,
    /// Standard deviation of pairwise distances
    pub distance_std: f32,
    /// Energy variance across population
    pub energy_variance: f32,
    /// Effective sample size (ESS) - accounts for correlation
    pub effective_sample_size: f32,
    /// Estimated number of distinct modes (clusters)
    pub estimated_modes: usize,
    /// Coverage: fraction of bounding box occupied
    pub coverage: f32,
    /// Dimension used for computation
    pub dim: usize,
}
