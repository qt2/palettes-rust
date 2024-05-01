mod renderer;
mod simulator;
mod types;

use std::{
    fs,
    time::{Duration, Instant},
};

use eframe::egui;
use simulator::{config::Config, Pedestrian, Simulator};

use crate::renderer::Renderer;
pub use crate::types::*;

fn main() {
    env_logger::init();

    #[cfg(feature = "gpu")]
    unsafe {
        hello();
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([960.0, 720.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("Pallets", options, Box::new(|cc| Box::new(App::new(cc)))).unwrap();
}

struct App {
    renderer: Renderer,
    simulator: Simulator,
    simulate_time: Duration,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let config = fs::read_to_string("cases/simple.toml").expect("case file not found");
        let config: Config = toml::from_str(&config).unwrap();

        App {
            renderer: Renderer::new(cc),
            simulator: Simulator::from_config(config),
            simulate_time: Duration::ZERO,
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let start = Instant::now();
        self.simulator.tick();
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

#[cfg(feature = "gpu")]
extern "C" {
    fn hello();
}
