pub mod config;
mod gate;
mod pedestrian;
mod wall;

use rayon::prelude::*;

use crate::{tick_pedestrians, RuntimeKind, Vec2};

pub use self::pedestrian::{Pedestrian, PedestriansT};
use self::{config::Config, gate::Gate};

#[derive(Debug, Default)]
pub struct Simulator {
    pub config: Config,
    pub gates: Vec<Gate>,
    pub pedestrians: Vec<Pedestrian>,
}

impl Simulator {
    pub fn from_config(config: Config) -> Self {
        fastrand::seed(0);

        let gates: Vec<Gate> = config
            .gates
            .iter()
            .map(|c| Gate { vertice: c.vertice })
            .collect();

        let pedestrians = config
            .pedestrians
            .iter()
            .enumerate()
            .flat_map(|(trip_id, p)| {
                let goal_x = gates[p.to].vertice[0].x;

                (0..p.spawn_count).map(move |_| {
                    let [r1, r2] = p.spawn_rect;
                    let [tx, ty] = [fastrand::f32(), fastrand::f32()];
                    let x = tx * r1.x + (1.0 - tx) * r2.x;
                    let y = ty * r1.y + (1.0 - ty) * r2.y;
                    let position = Vec2::new(x, y);
                    let goal = Vec2::new(goal_x, y);

                    Pedestrian {
                        position,
                        trip_id,
                        goal_id: p.to,
                        goal,
                        ..Default::default()
                    }
                })
            })
            .collect();

        Simulator {
            config,
            gates,
            pedestrians,
            ..Default::default()
        }
    }
}

impl Simulator {
    pub fn tick(&mut self, runtime: RuntimeKind) {
        // self.spawn_pedestrians();

        unsafe {
            let simulator = self as *mut Simulator as usize;

            match runtime {
                RuntimeKind::Single => {
                    self.pedestrians
                        .iter_mut()
                        .for_each(|p| p.determine_accel(&*(simulator as *mut Simulator)));
                }
                RuntimeKind::Multi => {
                    self.pedestrians
                        .par_iter_mut()
                        .for_each(|p| p.determine_accel(&*(simulator as *mut Simulator)));
                }
                RuntimeKind::GPU => {
                    #[cfg(feature = "gpu")]
                    {
                        let pedestrians = PedestriansT::from_pedestrians(&self.pedestrians);
                        tick_pedestrians(pedestrians, self.pedestrians.len());
                    }

                    #[cfg(not(feature = "gpu"))]
                    {
                        panic!("compile with `gpu` feature")
                    }
                }
            }
        }

        self.pedestrians.iter_mut().for_each(|p| p.walk());
    }

    // fn spawn_pedestrians(&mut self) {
    //     for (trip_id, config) in self.config.pedestrians.iter().enumerate() {
    //         let mut prob = config.frequency;
    //         loop {
    //             prob -= fastrand::f32();

    //             if prob < 0.0 {
    //                 break;
    //             }

    //             let start = &self.gates[config.from];
    //             let goal = &self.gates[config.to];

    //             let start = start.vertice[0].lerp(&start.vertice[1], fastrand::f32());
    //             let goal = goal.vertice[0].lerp(&goal.vertice[1], fastrand::f32());

    //             self.pedestrians.push(Pedestrian {
    //                 position: start,
    //                 trip_id,
    //                 goal,
    //                 velocity: (goal - start).normalize(),
    //                 ..Default::default()
    //             });
    //         }
    //     }
    // }
}
