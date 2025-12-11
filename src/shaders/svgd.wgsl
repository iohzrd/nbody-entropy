// SVGD (Stein Variational Gradient Descent) compute shader
// GPU-accelerated O(n²) pairwise kernel computation + Langevin dynamics

struct Particle {
    pos: array<f32, 8>,      // Parameters (up to 8D, use first `dim` elements)
    grad: array<f32, 8>,     // Accumulated gradient + repulsion
    energy: f32,             // Current loss value
    _pad: array<f32, 7>,     // Padding for alignment
}

struct Uniforms {
    particle_count: u32,
    dim: u32,                // Number of dimensions actually used
    gamma: f32,              // Friction coefficient
    temperature: f32,        // Boltzmann temperature
    repulsion_strength: f32, // SVGD repulsion strength
    kernel_bandwidth: f32,   // RBF kernel bandwidth h
    dt: f32,                 // Time step
    seed: u32,               // Random seed for noise
}

// Training data for neural net (y = w2 * tanh(w1 * x))
// Stored as x values, then y values
const TRAIN_X: array<f32, 10> = array<f32, 10>(
    -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5
);
const TRAIN_Y: array<f32, 10> = array<f32, 10>(
    -1.93, -1.79, -1.52, -0.93, 0.0, 0.93, 1.52, 1.79, 1.93, 1.97
);
const TRAIN_SIZE: u32 = 10u;

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> repulsion: array<Particle>; // Temp buffer for repulsion forces

// Simple hash for pseudo-random numbers
fn hash(seed: u32) -> u32 {
    var x = seed;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    x = x ^ (x >> 16u);
    return x;
}

// Generate uniform random in [0, 1)
fn rand(seed: u32) -> f32 {
    return f32(hash(seed) & 0xFFFFFFu) / 16777216.0;
}

// Box-Muller transform for Gaussian noise
fn randn(seed1: u32, seed2: u32) -> f32 {
    let u1 = max(rand(seed1), 0.0001);
    let u2 = rand(seed2);
    return sqrt(-2.0 * log(u1)) * cos(6.283185307 * u2);
}

// Neural net forward: y = w3*tanh(w1*x) + w4*tanh(w2*x) (2 hidden neurons)
// For 2D mode: y = w2 * tanh(w1 * x)
fn nn_forward_2d(w1: f32, w2: f32, x: f32) -> f32 {
    return w2 * tanh(w1 * x);
}

fn nn_forward_4d(w1: f32, w2: f32, w3: f32, w4: f32, x: f32) -> f32 {
    return w3 * tanh(w1 * x) + w4 * tanh(w2 * x);
}

// MSE loss for 2D neural net
fn nn_loss_2d(w1: f32, w2: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let pred = nn_forward_2d(w1, w2, TRAIN_X[i]);
        let err = pred - TRAIN_Y[i];
        sum = sum + err * err;
    }
    return sum / f32(TRAIN_SIZE);
}

// MSE loss for 4D neural net
fn nn_loss_4d(w1: f32, w2: f32, w3: f32, w4: f32) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let pred = nn_forward_4d(w1, w2, w3, w4, TRAIN_X[i]);
        let err = pred - TRAIN_Y[i];
        sum = sum + err * err;
    }
    return sum / f32(TRAIN_SIZE);
}

// Gradient of 2D neural net loss
fn nn_gradient_2d(w1: f32, w2: f32) -> vec2<f32> {
    var dw1 = 0.0;
    var dw2 = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let x = TRAIN_X[i];
        let z = w1 * x;
        let a = tanh(z);
        let pred = w2 * a;
        let err = pred - TRAIN_Y[i];
        let tanh_deriv = 1.0 - a * a;
        dw1 = dw1 + err * w2 * tanh_deriv * x;
        dw2 = dw2 + err * a;
    }
    let n = f32(TRAIN_SIZE);
    return vec2<f32>(2.0 * dw1 / n, 2.0 * dw2 / n);
}

// Gradient of 4D neural net loss
fn nn_gradient_4d(w1: f32, w2: f32, w3: f32, w4: f32) -> vec4<f32> {
    var dw1 = 0.0;
    var dw2 = 0.0;
    var dw3 = 0.0;
    var dw4 = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let x = TRAIN_X[i];
        let z1 = w1 * x;
        let z2 = w2 * x;
        let a1 = tanh(z1);
        let a2 = tanh(z2);
        let pred = w3 * a1 + w4 * a2;
        let err = pred - TRAIN_Y[i];
        let tanh_deriv1 = 1.0 - a1 * a1;
        let tanh_deriv2 = 1.0 - a2 * a2;
        dw1 = dw1 + err * w3 * tanh_deriv1 * x;
        dw2 = dw2 + err * w4 * tanh_deriv2 * x;
        dw3 = dw3 + err * a1;
        dw4 = dw4 + err * a2;
    }
    let n = f32(TRAIN_SIZE);
    return vec4<f32>(2.0 * dw1 / n, 2.0 * dw2 / n, 2.0 * dw3 / n, 2.0 * dw4 / n);
}

// 8D neural net: y = w5*tanh(w1*x) + w6*tanh(w2*x) + w7*tanh(w3*x) + w8*tanh(w4*x)
fn nn_forward_8d(w: ptr<function, Particle>, x: f32) -> f32 {
    return (*w).pos[4] * tanh((*w).pos[0] * x) +
           (*w).pos[5] * tanh((*w).pos[1] * x) +
           (*w).pos[6] * tanh((*w).pos[2] * x) +
           (*w).pos[7] * tanh((*w).pos[3] * x);
}

fn nn_loss_8d(w: ptr<function, Particle>) -> f32 {
    var sum = 0.0;
    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let pred = nn_forward_8d(w, TRAIN_X[i]);
        let err = pred - TRAIN_Y[i];
        sum = sum + err * err;
    }
    return sum / f32(TRAIN_SIZE);
}

fn nn_gradient_8d(w: ptr<function, Particle>) -> array<f32, 8> {
    var grad: array<f32, 8>;
    for (var d = 0u; d < 8u; d = d + 1u) {
        grad[d] = 0.0;
    }

    for (var i = 0u; i < TRAIN_SIZE; i = i + 1u) {
        let x = TRAIN_X[i];
        let a0 = tanh((*w).pos[0] * x);
        let a1 = tanh((*w).pos[1] * x);
        let a2 = tanh((*w).pos[2] * x);
        let a3 = tanh((*w).pos[3] * x);

        let pred = (*w).pos[4] * a0 + (*w).pos[5] * a1 + (*w).pos[6] * a2 + (*w).pos[7] * a3;
        let err = pred - TRAIN_Y[i];

        let td0 = 1.0 - a0 * a0;
        let td1 = 1.0 - a1 * a1;
        let td2 = 1.0 - a2 * a2;
        let td3 = 1.0 - a3 * a3;

        // Gradients w.r.t. input weights
        grad[0] = grad[0] + err * (*w).pos[4] * td0 * x;
        grad[1] = grad[1] + err * (*w).pos[5] * td1 * x;
        grad[2] = grad[2] + err * (*w).pos[6] * td2 * x;
        grad[3] = grad[3] + err * (*w).pos[7] * td3 * x;

        // Gradients w.r.t. output weights
        grad[4] = grad[4] + err * a0;
        grad[5] = grad[5] + err * a1;
        grad[6] = grad[6] + err * a2;
        grad[7] = grad[7] + err * a3;
    }

    let n = f32(TRAIN_SIZE);
    for (var d = 0u; d < 8u; d = d + 1u) {
        grad[d] = 2.0 * grad[d] / n;
    }
    return grad;
}

// RBF kernel: K(x,y) = exp(-||x-y||² / (2h²))
fn rbf_kernel(p1: ptr<function, Particle>, p2: ptr<function, Particle>) -> f32 {
    var dist_sq = 0.0;
    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        let diff = (*p1).pos[d] - (*p2).pos[d];
        dist_sq = dist_sq + diff * diff;
    }
    let h_sq = uniforms.kernel_bandwidth * uniforms.kernel_bandwidth;
    return exp(-dist_sq / (2.0 * h_sq));
}

// Pass 1: Compute O(n²) pairwise repulsion forces
@compute @workgroup_size(64)
fn compute_repulsion(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.particle_count {
        return;
    }

    var p_i = particles[idx];
    let h_sq = uniforms.kernel_bandwidth * uniforms.kernel_bandwidth;

    // Zero out repulsion gradient
    for (var d = 0u; d < 8u; d = d + 1u) {
        repulsion[idx].grad[d] = 0.0;
    }

    // Sum repulsion from all other particles
    for (var j = 0u; j < uniforms.particle_count; j = j + 1u) {
        if j == idx {
            continue;
        }
        var p_j = particles[j];

        // RBF kernel
        var dist_sq = 0.0;
        for (var d = 0u; d < uniforms.dim; d = d + 1u) {
            let diff = p_i.pos[d] - p_j.pos[d];
            dist_sq = dist_sq + diff * diff;
        }
        let k = exp(-dist_sq / (2.0 * h_sq));

        // Kernel gradient: -K * (x_i - x_j) / h²
        for (var d = 0u; d < uniforms.dim; d = d + 1u) {
            let diff = p_i.pos[d] - p_j.pos[d];
            repulsion[idx].grad[d] = repulsion[idx].grad[d] - k * diff / h_sq;
        }
    }

    // Normalize and scale
    let n = f32(uniforms.particle_count);
    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        repulsion[idx].grad[d] = repulsion[idx].grad[d] * uniforms.repulsion_strength / n;
    }
}

// Pass 2: Update particles with Langevin dynamics
@compute @workgroup_size(64)
fn update_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if idx >= uniforms.particle_count {
        return;
    }

    var p = particles[idx];

    // Compute loss gradient based on dimension
    if uniforms.dim == 2u {
        let loss_grad = nn_gradient_2d(p.pos[0], p.pos[1]);
        p.grad[0] = loss_grad.x;
        p.grad[1] = loss_grad.y;
        p.energy = nn_loss_2d(p.pos[0], p.pos[1]);
    } else if uniforms.dim == 4u {
        let loss_grad = nn_gradient_4d(p.pos[0], p.pos[1], p.pos[2], p.pos[3]);
        p.grad[0] = loss_grad.x;
        p.grad[1] = loss_grad.y;
        p.grad[2] = loss_grad.z;
        p.grad[3] = loss_grad.w;
        p.energy = nn_loss_4d(p.pos[0], p.pos[1], p.pos[2], p.pos[3]);
    } else if uniforms.dim == 8u {
        let loss_grad = nn_gradient_8d(&p);
        for (var d = 0u; d < 8u; d = d + 1u) {
            p.grad[d] = loss_grad[d];
        }
        p.energy = nn_loss_8d(&p);
    } else {
        // Default to 2D
        let loss_grad = nn_gradient_2d(p.pos[0], p.pos[1]);
        p.grad[0] = loss_grad.x;
        p.grad[1] = loss_grad.y;
        p.energy = nn_loss_2d(p.pos[0], p.pos[1]);
    }

    // Langevin update: dx = (-γ∇E + repulsion) * dt + √(2γT) * dW
    let noise_scale = sqrt(2.0 * uniforms.gamma * uniforms.temperature * uniforms.dt);

    for (var d = 0u; d < uniforms.dim; d = d + 1u) {
        // Generate noise
        let seed1 = uniforms.seed + idx * 17u + d * 31u;
        let seed2 = uniforms.seed + idx * 37u + d * 53u + 12345u;
        let noise = randn(seed1, seed2);

        // Update position
        let grad_term = -uniforms.gamma * p.grad[d];
        let repulsion_term = repulsion[idx].grad[d];
        p.pos[d] = p.pos[d] + (grad_term + repulsion_term) * uniforms.dt + noise_scale * noise;

        // Clamp to domain [-4, 4]
        p.pos[d] = clamp(p.pos[d], -4.0, 4.0);
    }

    particles[idx] = p;
}
