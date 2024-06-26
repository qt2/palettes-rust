use serde::Deserialize;

use crate::Vec2;

#[derive(Debug, Default, Deserialize)]
pub struct Config {
    pub gates: Vec<GateConfig>,
    pub pedestrians: Vec<PedestrianConfig>,
}

#[derive(Debug, Deserialize)]
pub struct GateConfig {
    pub vertice: [Vec2; 2],
}

#[derive(Debug, Deserialize)]
pub struct PedestrianConfig {
    pub from: usize,
    pub to: usize,
    pub frequency: f32,
}
