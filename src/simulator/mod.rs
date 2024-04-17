pub mod config;
mod gate;
mod pedestrian;
mod wall;

use serde::Deserialize;

pub use self::pedestrian::Pedestrian;
use self::{config::Config, gate::Gate};

#[derive(Debug, Default)]
pub struct Simulator {
    pub config: Config,
    pub gates: Vec<Gate>,
    pub pedestrians: Vec<Pedestrian>,
}

impl Simulator {
    pub fn from_config(config: Config) -> Self {
        let gates = config
            .gates
            .iter()
            .map(|c| Gate { vertice: c.vertice })
            .collect();

        Simulator {
            config,
            gates,
            ..Default::default()
        }
    }
}

impl Simulator {
    pub fn tick(&mut self) {
        for config in &self.config.pedestrians {
            let mut prob = config.frequency;
            loop {
                prob -= fastrand::f32();

                if prob < 0.0 {
                    break;
                }

                let start = &self.gates[config.from];
                let goal = &self.gates[config.to];

                let start = start.vertice[0].lerp(&start.vertice[1], fastrand::f32());
                let goal = goal.vertice[0].lerp(&goal.vertice[1], fastrand::f32());

                self.pedestrians.push(Pedestrian {
                    position: start,
                    goal,
                    velocity: (goal - start).normalize(),
                    ..Default::default()
                });
            }
        }

        self.pedestrians.iter_mut().for_each(|p| p.walk());
    }
}
