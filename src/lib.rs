pub mod gpu;
pub mod gpu_svgd;
pub mod thermodynamic;

pub use gpu::GpuNbodyEntropy;
pub use gpu_svgd::GpuSvgd;
pub use thermodynamic::{ThermodynamicSystem, ThermodynamicMode, ThermodynamicParticle, ThermodynamicStats};

pub const DEFAULT_PARTICLE_COUNT: usize = 64;
pub const ATTRACTOR_COUNT: usize = 8;
pub const FOLLOWER_COUNT: usize = DEFAULT_PARTICLE_COUNT - ATTRACTOR_COUNT;
