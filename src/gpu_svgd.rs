//! GPU-accelerated SVGD for ML optimization
//!
//! Uses wgpu compute shaders for O(n²) pairwise kernel computation.
//! Supports higher-dimensional parameter spaces with 2D projection for visualization.

use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Maximum parameter dimensions supported
pub const MAX_DIM: usize = 8;

/// GPU particle representation (matches svgd.wgsl Particle struct)
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct GpuParticle {
    pub pos: [f32; MAX_DIM],
    pub grad: [f32; MAX_DIM],
    pub energy: f32,
    pub _pad: [f32; 7],
}

/// GPU uniforms for SVGD
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SvgdUniforms {
    particle_count: u32,
    dim: u32,
    gamma: f32,
    temperature: f32,
    repulsion_strength: f32,
    kernel_bandwidth: f32,
    dt: f32,
    seed: u32,
}

/// GPU-accelerated SVGD sampler
pub struct GpuSvgd {
    device: wgpu::Device,
    queue: wgpu::Queue,
    particle_buffer: wgpu::Buffer,
    repulsion_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
    staging_buffer: wgpu::Buffer,
    repulsion_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    particle_count: usize,
    dim: usize,
    step: u32,
}

impl GpuSvgd {
    pub fn new(
        particle_count: usize,
        dim: usize,
        gamma: f32,
        temperature: f32,
        repulsion_strength: f32,
        kernel_bandwidth: f32,
        dt: f32,
    ) -> Self {
        assert!(dim <= MAX_DIM, "Dimension must be <= {}", MAX_DIM);

        // Initialize particles randomly
        let mut rng_state = 42u64;
        let mut particles = Vec::with_capacity(particle_count);

        for _ in 0..particle_count {
            let mut pos = [0.0f32; MAX_DIM];
            for d in 0..dim {
                rng_state ^= rng_state << 13;
                rng_state ^= rng_state >> 7;
                rng_state ^= rng_state << 17;
                // Initialize in [-4, 4]
                pos[d] = -4.0 + (rng_state & 0xFFFF) as f32 / 65535.0 * 8.0;
            }
            particles.push(GpuParticle {
                pos,
                grad: [0.0; MAX_DIM],
                energy: 0.0,
                _pad: [0.0; 7],
            });
        }

        // Create wgpu instance
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .expect("No GPU adapter");

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("SVGD Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::Performance,
            ..Default::default()
        }))
        .expect("Failed to create device");

        let particle_size = std::mem::size_of::<GpuParticle>() * particle_count;

        // Create buffers
        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particle Buffer"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let repulsion_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Repulsion Buffer"),
            size: particle_size as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let uniforms = SvgdUniforms {
            particle_count: particle_count as u32,
            dim: dim as u32,
            gamma,
            temperature,
            repulsion_strength,
            kernel_bandwidth,
            dt,
            seed: 12345,
        };

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: particle_size as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SVGD Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/svgd.wgsl").into()),
        });

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SVGD Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SVGD Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: repulsion_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SVGD Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let repulsion_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Repulsion Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_repulsion"),
            compilation_options: Default::default(),
            cache: None,
        });

        let update_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Update Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("update_particles"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            device,
            queue,
            particle_buffer,
            repulsion_buffer,
            uniform_buffer,
            staging_buffer,
            repulsion_pipeline,
            update_pipeline,
            bind_group,
            particle_count,
            dim,
            step: 0,
        }
    }

    /// Run one SVGD step on GPU
    pub fn step(&mut self) {
        // Update seed for random noise
        self.step += 1;
        let seed_update = [self.step];
        self.queue.write_buffer(
            &self.uniform_buffer,
            28, // Offset to seed field
            bytemuck::cast_slice(&seed_update),
        );

        let workgroups = ((self.particle_count as u32) + 63) / 64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SVGD Encoder"),
        });

        // Pass 1: Compute repulsion forces
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Repulsion Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.repulsion_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        // Pass 2: Update particles
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Update Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.update_pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.queue.submit(Some(encoder.finish()));
    }

    /// Read particles back from GPU
    pub fn read_particles(&self) -> Vec<GpuParticle> {
        let size = (std::mem::size_of::<GpuParticle>() * self.particle_count) as u64;

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &self.staging_buffer, 0, size);

        self.queue.submit(Some(encoder.finish()));

        let slice = self.staging_buffer.slice(..size);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        self.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        rx.recv().unwrap().expect("Map failed");

        let data = slice.get_mapped_range();
        let particles: Vec<GpuParticle> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        self.staging_buffer.unmap();

        particles
    }

    pub fn particle_count(&self) -> usize {
        self.particle_count
    }

    pub fn dim(&self) -> usize {
        self.dim
    }
}
