//! SVGD-style particle sampler for ML optimization
//!
//! Tiny neural net: y = w2 * tanh(w1 * x)
//! Two parameters [w1, w2] create non-convex loss surface.
//! Symmetry: (w1, w2) and (-w1, -w2) are equivalent solutions.
//!
//! Run with: cargo run --release --features viz --bin nbody-viz

use iced::mouse;
use iced::widget::{canvas, container};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Subscription, Theme};

// Sampler constants
const SAMPLER_COUNT: usize = 50;
const GAMMA: f32 = 0.5;              // Friction coefficient for Langevin dynamics
const TEMPERATURE: f32 = 0.05;       // Boltzmann temperature
const REPULSION_STRENGTH: f32 = 0.2; // SVGD-style repulsion between particles
const KERNEL_BANDWIDTH: f32 = 0.3;   // RBF kernel bandwidth (h)

// Simulation constants
const DT: f32 = 0.01;                // Time step
const DOMAIN_MIN: f32 = -4.0;        // Parameter space for [w1, w2]
const DOMAIN_MAX: f32 = 4.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

// Training data for: y = 2.0 * tanh(1.5 * x)
// Two equivalent optima: (w1=1.5, w2=2.0) and (w1=-1.5, w2=-2.0)
const TRAIN_DATA: [(f32, f32); 10] = [
    (-2.0, -1.93),  // 2.0 * tanh(-3.0) ≈ -1.99
    (-1.5, -1.79),  // 2.0 * tanh(-2.25) ≈ -1.98
    (-1.0, -1.52),  // 2.0 * tanh(-1.5) ≈ -1.82
    (-0.5, -0.93),  // 2.0 * tanh(-0.75) ≈ -1.22
    (0.0, 0.0),     // 2.0 * tanh(0) = 0
    (0.5, 0.93),
    (1.0, 1.52),
    (1.5, 1.79),
    (2.0, 1.93),
    (2.5, 1.97),
];
// True parameters: w1=1.5, w2=2.0 (or w1=-1.5, w2=-2.0)

fn main() -> iced::Result {
    iced::application(App::default, update, view)
        .subscription(subscription)
        .title("SVGD Tiny Neural Net")
        .run()
}

#[derive(Default)]
struct App {
    particles: ParticleSystem,
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            app.particles.step_physics();
            app.particles.request_redraw();
        }
    }
}

fn view(app: &App) -> Element<'_, Message> {
    container(
        canvas(&app.particles)
            .width(Length::Fill)
            .height(Length::Fill),
    )
    .width(Length::Fill)
    .height(Length::Fill)
    .style(|_| container::Style {
        background: Some(iced::Background::Color(Color::from_rgb(0.05, 0.05, 0.1))),
        ..Default::default()
    })
    .into()
}

fn subscription(_app: &App) -> Subscription<Message> {
    iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick)
}

struct Particle {
    pos: [f32; 2],  // Position in domain [DOMAIN_MIN, DOMAIN_MAX]
    vel: [f32; 2],  // Velocity (momentum for Langevin dynamics)
}

/// Neural net forward pass: y = w2 * tanh(w1 * x)
#[inline]
fn nn_forward(w1: f32, w2: f32, x: f32) -> f32 {
    w2 * (w1 * x).tanh()
}

/// MSE loss for neural net
/// E(w1, w2) = (1/n) Σ (w2 * tanh(w1 * x_i) - y_i)²
#[inline]
fn nn_loss(w1: f32, w2: f32) -> f32 {
    let mut sum = 0.0;
    for (x, y) in TRAIN_DATA.iter() {
        let pred = nn_forward(w1, w2, *x);
        let err = pred - y;
        sum += err * err;
    }
    sum / TRAIN_DATA.len() as f32
}

/// Gradient of neural net loss w.r.t. [w1, w2]
/// ∂E/∂w1 = (2/n) Σ (pred - y) * w2 * (1 - tanh²(w1*x)) * x
/// ∂E/∂w2 = (2/n) Σ (pred - y) * tanh(w1*x)
#[inline]
fn nn_gradient(w1: f32, w2: f32) -> [f32; 2] {
    let mut dw1 = 0.0;
    let mut dw2 = 0.0;
    for (x, y) in TRAIN_DATA.iter() {
        let z = w1 * x;           // pre-activation
        let a = z.tanh();         // activation
        let pred = w2 * a;        // output
        let err = pred - y;       // error

        // tanh derivative: 1 - tanh²(z)
        let tanh_deriv = 1.0 - a * a;

        dw1 += err * w2 * tanh_deriv * x;
        dw2 += err * a;
    }
    let n = TRAIN_DATA.len() as f32;
    [2.0 * dw1 / n, 2.0 * dw2 / n]
}

/// RBF kernel: K(x,y) = exp(-||x-y||² / (2h²))
#[inline]
fn rbf_kernel(dx: f32, dy: f32) -> f32 {
    let dist_sq = dx * dx + dy * dy;
    (-dist_sq / (2.0 * KERNEL_BANDWIDTH * KERNEL_BANDWIDTH)).exp()
}

/// Gradient of RBF kernel with respect to first argument
/// ∇_x K(x,y) = -K(x,y) * (x-y) / h²
#[inline]
fn rbf_kernel_gradient(dx: f32, dy: f32) -> [f32; 2] {
    let k = rbf_kernel(dx, dy);
    let h_sq = KERNEL_BANDWIDTH * KERNEL_BANDWIDTH;
    [-k * dx / h_sq, -k * dy / h_sq]
}

struct ParticleSystem {
    particles: Vec<Particle>,
    cache: canvas::Cache,
    rng_state: u64,
}

impl Default for ParticleSystem {
    fn default() -> Self {
        Self::new(42)
    }
}

/// Clamp position to domain
#[inline]
fn clamp_to_domain(x: f32) -> f32 {
    x.clamp(DOMAIN_MIN, DOMAIN_MAX)
}

impl ParticleSystem {
    fn new(seed: u64) -> Self {
        let mut state = seed;
        let mut particles = Vec::with_capacity(SAMPLER_COUNT);

        for _ in 0..SAMPLER_COUNT {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;

            // Initialize uniformly across domain
            let pos = [
                DOMAIN_MIN + (state & 0xFFFF) as f32 / 65535.0 * DOMAIN_SIZE,
                DOMAIN_MIN + ((state >> 16) & 0xFFFF) as f32 / 65535.0 * DOMAIN_SIZE,
            ];

            particles.push(Particle {
                pos,
                vel: [0.0, 0.0],
            });
        }

        Self {
            particles,
            cache: canvas::Cache::new(),
            rng_state: state,
        }
    }

    /// Simple xorshift random number generator
    fn rand(&mut self) -> f32 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        (self.rng_state & 0xFFFFFF) as f32 / 16777215.0
    }

    /// Box-Muller transform for Gaussian noise
    fn randn(&mut self) -> f32 {
        let u1 = self.rand().max(1e-10);
        let u2 = self.rand();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    fn request_redraw(&self) {
        self.cache.clear();
    }

    /// Step the Langevin dynamics with SVGD repulsion
    fn step_physics(&mut self) {
        let n = self.particles.len();

        // Collect current positions for O(n²) repulsion calculation
        let positions: Vec<[f32; 2]> = self.particles.iter().map(|p| p.pos).collect();

        // Compute SVGD repulsion forces for all particles
        let mut repulsion_forces: Vec<[f32; 2]> = vec![[0.0, 0.0]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let dx = positions[i][0] - positions[j][0];
                let dy = positions[i][1] - positions[j][1];

                // RBF kernel gradient gives repulsion direction
                let grad = rbf_kernel_gradient(dx, dy);
                repulsion_forces[i][0] += grad[0];
                repulsion_forces[i][1] += grad[1];
            }
            // Normalize by number of particles
            repulsion_forces[i][0] *= REPULSION_STRENGTH / n as f32;
            repulsion_forces[i][1] *= REPULSION_STRENGTH / n as f32;
        }

        // Update each particle with Langevin dynamics
        // dx = -γ∇E(x)dt + repulsion + √(2γT)dW
        let noise_scale = (2.0 * GAMMA * TEMPERATURE * DT).sqrt();

        for i in 0..n {
            // Generate noise before borrowing particle
            let noise_x = self.randn();
            let noise_y = self.randn();

            let p = &mut self.particles[i];

            // Energy gradient (drives particles toward low loss)
            let grad = nn_gradient(p.pos[0], p.pos[1]);

            // Langevin update with SVGD repulsion
            // Overdamped Langevin: we directly update position (no momentum)
            p.pos[0] += (-GAMMA * grad[0] + repulsion_forces[i][0]) * DT + noise_scale * noise_x;
            p.pos[1] += (-GAMMA * grad[1] + repulsion_forces[i][1]) * DT + noise_scale * noise_y;

            // Clamp to domain (soft boundary)
            p.pos[0] = clamp_to_domain(p.pos[0]);
            p.pos[1] = clamp_to_domain(p.pos[1]);
        }
    }

    /// Map particle position from domain to screen coordinates
    fn get_position(&self, p: &Particle, bounds: Size) -> Point {
        let norm_x = (p.pos[0] - DOMAIN_MIN) / DOMAIN_SIZE;
        let norm_y = (p.pos[1] - DOMAIN_MIN) / DOMAIN_SIZE;
        Point::new(norm_x * bounds.width, norm_y * bounds.height)
    }

    /// Get loss at a particle's position (w1, w2 parameters)
    fn get_energy(&self, p: &Particle) -> f32 {
        nn_loss(p.pos[0], p.pos[1])
    }
}

impl canvas::Program<Message> for ParticleSystem {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry<Renderer>> {
        let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
            let size = bounds.size();

            // Draw grid lines for parameter space
            let grid_color = Color::from_rgba(0.2, 0.3, 0.4, 0.3);
            for i in -3..=3 {
                let x = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

                // Vertical line (weight values)
                let vline = canvas::Path::line(
                    Point::new(x, 0.0),
                    Point::new(x, size.height),
                );
                frame.stroke(&vline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));

                // Horizontal line (bias values)
                let hline = canvas::Path::line(
                    Point::new(0.0, y),
                    Point::new(size.width, y),
                );
                frame.stroke(&hline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));
            }

            // Highlight both equivalent optima
            // Optimum 1: (w1=1.5, w2=2.0)
            let opt1_x = ((1.5 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let opt1_y = ((2.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let optimum1 = canvas::Path::circle(Point::new(opt1_x, opt1_y), 12.0);
            frame.stroke(
                &optimum1,
                canvas::Stroke::default()
                    .with_color(Color::from_rgba(0.0, 1.0, 0.5, 0.8))
                    .with_width(3.0),
            );
            let opt1_dot = canvas::Path::circle(Point::new(opt1_x, opt1_y), 3.0);
            frame.fill(&opt1_dot, Color::from_rgb(0.0, 1.0, 0.5));

            // Optimum 2: (w1=-1.5, w2=-2.0) - symmetric solution
            let opt2_x = ((-1.5 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
            let opt2_y = ((-2.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
            let optimum2 = canvas::Path::circle(Point::new(opt2_x, opt2_y), 12.0);
            frame.stroke(
                &optimum2,
                canvas::Stroke::default()
                    .with_color(Color::from_rgba(0.0, 1.0, 0.5, 0.8))
                    .with_width(3.0),
            );
            let opt2_dot = canvas::Path::circle(Point::new(opt2_x, opt2_y), 3.0);
            frame.fill(&opt2_dot, Color::from_rgb(0.0, 1.0, 0.5));

            // Draw particles colored by loss
            // Lower loss = greener, higher loss = redder
            let max_energy = 10.0; // MSE scale

            for p in &self.particles {
                let pos = self.get_position(p, size);
                let energy = self.get_energy(p);

                // Color: green (low energy) to red (high energy)
                let t = (energy / max_energy).clamp(0.0, 1.0);
                let color = Color::from_rgb(
                    0.2 + 0.8 * t,      // Red increases with energy
                    0.8 * (1.0 - t),    // Green decreases with energy
                    0.3,                 // Constant blue
                );

                let circle = canvas::Path::circle(pos, 4.0);
                frame.fill(&circle, color);
            }

            // Draw a small dot at each particle position for better visibility
            for p in &self.particles {
                let pos = self.get_position(p, size);
                let dot = canvas::Path::circle(pos, 1.5);
                frame.fill(&dot, Color::WHITE);
            }
        });

        vec![geometry]
    }
}
