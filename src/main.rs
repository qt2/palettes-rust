mod renderer;

use eframe::egui;

use crate::renderer::Renderer;

fn main() {
    env_logger::init();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([320.0, 240.0]),
        renderer: eframe::Renderer::Wgpu,
        ..Default::default()
    };
    eframe::run_native("Pallets", options, Box::new(|cc| Box::new(App::new(cc)))).unwrap();
}

struct App {
    renderer: Renderer,
}

impl App {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        App {
            renderer: Renderer::new(cc),
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("My egui Application");
            egui::Frame::canvas(ui.style()).show(ui, |ui| {
                self.renderer.draw_canvas(ui);
            });
        });
    }
}
