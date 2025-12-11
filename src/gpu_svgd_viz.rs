//! GPU-accelerated SVGD visualization
//!
//! Uses wgpu compute shaders for O(n²) pairwise computation.
//! Scales to 1000+ particles easily.
//!
//! Run with: cargo run --release --features "viz gpu" --bin gpu-svgd-viz

use iced::mouse;
use iced::widget::{canvas, container};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Subscription, Theme};
use nbody_entropy::gpu_svgd::{GpuParticle, GpuSvgd};

// Scalable particle count - GPU can handle many more!
const PARTICLE_COUNT: usize = 1000;
const DIM: usize = 8;  // 8D parameter space (4-hidden-neuron net)

// SVGD parameters (tuned for 8D - higher bandwidth, lower repulsion)
const GAMMA: f32 = 0.3;
const TEMPERATURE: f32 = 0.02;
const REPULSION_STRENGTH: f32 = 0.1;  // Lower for 8D to avoid over-spreading
const KERNEL_BANDWIDTH: f32 = 0.8;  // Larger bandwidth for higher dims
const DT: f32 = 0.01;

// Domain
const DOMAIN_MIN: f32 = -4.0;
const DOMAIN_MAX: f32 = 4.0;
const DOMAIN_SIZE: f32 = DOMAIN_MAX - DOMAIN_MIN;

fn main() -> iced::Result {
    iced::application(App::new, update, view)
        .subscription(subscription)
        .title("GPU SVGD - 8D Neural Net (1000 particles)")
        .run()
}

struct App {
    gpu_svgd: GpuSvgd,
    particles: Vec<GpuParticle>,
    cache: canvas::Cache,
    step_count: u32,
}

impl App {
    fn new() -> Self {
        let gpu_svgd = GpuSvgd::new(
            PARTICLE_COUNT,
            DIM,
            GAMMA,
            TEMPERATURE,
            REPULSION_STRENGTH,
            KERNEL_BANDWIDTH,
            DT,
        );
        let particles = gpu_svgd.read_particles();
        Self {
            gpu_svgd,
            particles,
            cache: canvas::Cache::new(),
            step_count: 0,
        }
    }
}

#[derive(Debug, Clone)]
enum Message {
    Tick,
}

fn update(app: &mut App, message: Message) {
    match message {
        Message::Tick => {
            // Run multiple GPU steps per frame for faster convergence
            for _ in 0..2 {
                app.gpu_svgd.step();
            }
            app.step_count += 2;

            // Read back particles every frame for visualization
            app.particles = app.gpu_svgd.read_particles();
            app.cache.clear();
        }
    }
}

fn view(app: &App) -> Element<'_, Message> {
    container(
        canvas(app)
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

impl canvas::Program<Message> for App {
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

            // Draw grid
            let grid_color = Color::from_rgba(0.2, 0.3, 0.4, 0.3);
            for i in -4..=4 {
                let x = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((i as f32 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;

                let vline = canvas::Path::line(Point::new(x, 0.0), Point::new(x, size.height));
                frame.stroke(&vline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));

                let hline = canvas::Path::line(Point::new(0.0, y), Point::new(size.width, y));
                frame.stroke(&hline, canvas::Stroke::default().with_color(grid_color).with_width(0.5));
            }

            // For 4D: Show origin as reference point
            // For 2D: Show both optima
            if DIM == 2 {
                for (w1, w2) in [(1.5, 2.0), (-1.5, -2.0)] {
                    let opt_x = ((w1 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                    let opt_y = ((w2 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                    let opt = canvas::Path::circle(Point::new(opt_x, opt_y), 12.0);
                    frame.stroke(
                        &opt,
                        canvas::Stroke::default()
                            .with_color(Color::from_rgba(0.0, 1.0, 0.5, 0.8))
                            .with_width(3.0),
                    );
                }
            } else {
                // For higher dims, just show origin
                let origin_x = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let origin_y = ((0.0 - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                let origin = canvas::Path::circle(Point::new(origin_x, origin_y), 8.0);
                frame.stroke(
                    &origin,
                    canvas::Stroke::default()
                        .with_color(Color::from_rgba(0.5, 0.5, 0.5, 0.5))
                        .with_width(2.0),
                );
            }

            // Draw particles
            let max_energy = 5.0;
            for p in &self.particles {
                let x = ((p.pos[0] - DOMAIN_MIN) / DOMAIN_SIZE) * size.width;
                let y = ((p.pos[1] - DOMAIN_MIN) / DOMAIN_SIZE) * size.height;
                let pos = Point::new(x, y);

                // Color by energy
                let t = (p.energy / max_energy).clamp(0.0, 1.0);
                let color = Color::from_rgb(
                    0.2 + 0.8 * t,
                    0.8 * (1.0 - t),
                    0.3,
                );

                // Smaller particles for large counts
                let radius = if PARTICLE_COUNT > 1000 { 1.5 } else if PARTICLE_COUNT > 200 { 2.0 } else { 4.0 };
                let circle = canvas::Path::circle(pos, radius);
                frame.fill(&circle, color);
            }

            // Draw loss histogram in bottom-right corner
            let hist_width = 200.0;
            let hist_height = 120.0;
            let hist_margin = 20.0;
            let hist_x = size.width - hist_width - hist_margin;
            let hist_y = size.height - hist_height - hist_margin;
            let num_bins = 20;
            let bin_width = hist_width / num_bins as f32;

            // Compute histogram bins (loss range 0 to max_energy)
            let mut bins = vec![0usize; num_bins];
            for p in &self.particles {
                let bin_idx = ((p.energy / max_energy) * (num_bins - 1) as f32).clamp(0.0, (num_bins - 1) as f32) as usize;
                bins[bin_idx] += 1;
            }
            let max_count = bins.iter().max().copied().unwrap_or(1).max(1);

            // Draw histogram background
            let bg = canvas::Path::rectangle(
                Point::new(hist_x - 5.0, hist_y - 5.0),
                iced::Size::new(hist_width + 10.0, hist_height + 10.0),
            );
            frame.fill(&bg, Color::from_rgba(0.0, 0.0, 0.0, 0.7));

            // Draw histogram bars
            for (i, &count) in bins.iter().enumerate() {
                let bar_height = (count as f32 / max_count as f32) * hist_height;
                let bar_x = hist_x + i as f32 * bin_width;
                let bar_y = hist_y + hist_height - bar_height;

                // Color gradient: green (low loss) -> red (high loss)
                let t = i as f32 / (num_bins - 1) as f32;
                let color = Color::from_rgba(
                    0.2 + 0.8 * t,
                    0.8 * (1.0 - t),
                    0.3,
                    0.9,
                );

                let bar = canvas::Path::rectangle(
                    Point::new(bar_x, bar_y),
                    iced::Size::new(bin_width - 1.0, bar_height),
                );
                frame.fill(&bar, color);
            }

            // Draw histogram border
            let border = canvas::Path::rectangle(
                Point::new(hist_x, hist_y),
                iced::Size::new(hist_width, hist_height),
            );
            frame.stroke(
                &border,
                canvas::Stroke::default()
                    .with_color(Color::from_rgba(0.5, 0.5, 0.5, 0.8))
                    .with_width(1.0),
            );

            // Draw axis lines
            let x_axis = canvas::Path::line(
                Point::new(hist_x, hist_y + hist_height),
                Point::new(hist_x + hist_width, hist_y + hist_height),
            );
            frame.stroke(
                &x_axis,
                canvas::Stroke::default()
                    .with_color(Color::from_rgba(0.7, 0.7, 0.7, 0.8))
                    .with_width(1.0),
            );

            // Draw tick marks for loss values
            for i in 0..=4 {
                let tick_x = hist_x + (i as f32 / 4.0) * hist_width;
                let tick = canvas::Path::line(
                    Point::new(tick_x, hist_y + hist_height),
                    Point::new(tick_x, hist_y + hist_height + 5.0),
                );
                frame.stroke(
                    &tick,
                    canvas::Stroke::default()
                        .with_color(Color::from_rgba(0.7, 0.7, 0.7, 0.8))
                        .with_width(1.0),
                );
            }

            // Compute and display stats via marker positions
            let total: f32 = self.particles.iter().map(|p| p.energy).sum();
            let mean_loss = total / self.particles.len() as f32;
            let min_loss = self.particles.iter().map(|p| p.energy).fold(f32::MAX, f32::min);

            // Draw mean loss marker (vertical line on histogram)
            let mean_x = hist_x + (mean_loss / max_energy).clamp(0.0, 1.0) * hist_width;
            let mean_line = canvas::Path::line(
                Point::new(mean_x, hist_y),
                Point::new(mean_x, hist_y + hist_height),
            );
            frame.stroke(
                &mean_line,
                canvas::Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 0.0, 0.9))
                    .with_width(2.0),
            );

            // Draw min loss marker
            let min_x = hist_x + (min_loss / max_energy).clamp(0.0, 1.0) * hist_width;
            let min_marker = canvas::Path::circle(Point::new(min_x, hist_y + hist_height - 10.0), 4.0);
            frame.fill(&min_marker, Color::from_rgba(0.0, 1.0, 1.0, 0.9));
        });

        vec![geometry]
    }
}
