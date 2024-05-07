pub mod renderer;
pub mod simulator;
pub mod types;

use std::{
    env, fs,
    time::{Duration, Instant},
};

use eframe::egui;
use simulator::{config::Config, PedestriansT, Simulator};

use crate::renderer::Renderer;
pub use crate::types::*;

fn main() {
    env_logger::init();

    #[cfg(feature = "gpu")]
    unsafe {
        hello();
    }

    #[cfg(not(feature = "headless"))]
    {
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 720.0]),
            renderer: eframe::Renderer::Wgpu,
            ..Default::default()
        };
        eframe::run_native("Pallets", options, Box::new(|cc| Box::new(App::new(cc)))).unwrap();
    }

    #[cfg(feature = "headless")]
    {
        let runtime = env::args().skip(1).next().unwrap_or("single".into());
        let runtime = RuntimeKind::from(runtime);

        let config = fs::read_to_string("cases/simple.toml").expect("case file not found");
        let config: Config = toml::from_str(&config).unwrap();

        let mut simulator = Simulator::from_config(config);

        let mut cum_duration = Duration::ZERO;
        loop {
            let start = Instant::now();
            simulator.tick(runtime);
            let duration = Instant::now() - start;
            cum_duration += duration;

            if cum_duration > Duration::from_secs(1) {
                println!("Calculation time per frame: {:.4}s", duration.as_secs_f64());
                cum_duration = Duration::ZERO;
            }
        }
    }
}

struct App {
    renderer: Renderer,
    simulator: Simulator,
    simulate_time: Duration,
    runtime: RuntimeKind,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let runtime = env::args().skip(1).next().unwrap_or("single".into());
        let runtime = RuntimeKind::from(runtime);

        let config = fs::read_to_string("cases/simple.toml").expect("case file not found");
        let config: Config = toml::from_str(&config).unwrap();

        App {
            renderer: Renderer::new(cc),
            simulator: Simulator::from_config(config),
            simulate_time: Duration::ZERO,
            runtime,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let start = Instant::now();
        self.simulator.tick(self.runtime);
        let duration = Instant::now() - start;
        self.simulate_time = (self.simulate_time * 15 + duration) / 16;

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("palettes-rust");
            ui.label(format!(
                "Number of pedestrians: {}",
                self.simulator.pedestrians.len(),
            ));
            ui.label(format!(
                "Calculation time per frame: {:.4}s",
                self.simulate_time.as_secs_f64()
            ));
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.renderer.draw_canvas(ui, ctx, &self.simulator);
            });
        });

        ctx.request_repaint();
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum RuntimeKind {
    #[default]
    Single,
    Multi,
    GPU,
}

impl From<String> for RuntimeKind {
    fn from(value: String) -> Self {
        match value.to_lowercase().as_str() {
            "single" => RuntimeKind::Single,
            "multi" => RuntimeKind::Multi,
            "gpu" => RuntimeKind::GPU,
            _ => panic!("unsupported runtime kind"),
        }
    }
}

#[cfg(feature = "gpu")]
extern "C" {
    fn hello();

    pub fn tick_pedestrians(pedestrians: PedestriansT, n: libc::size_t);
}
