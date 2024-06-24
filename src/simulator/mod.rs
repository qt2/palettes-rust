pub mod config;
mod gate;
mod pedestrian;
mod wall;

use std::{borrow::BorrowMut, time::Instant};

use rayon::prelude::*;

#[cfg(feature = "gpu")]
use self::pedestrian::tick_pedestrians;

pub use self::pedestrian::{Pedestrian, PedestriansT};
use self::{config::Config, gate::Gate};
use crate::{RuntimeKind, Vec2};

#[derive(Debug, Default)]
pub struct Simulator {
    pub config: Config,
    pub gates: Vec<Gate>,
    pub pedestrians: Vec<Pedestrian>,
    pub neighbor_lists: Vec<Vec<usize>>,
    pub step: i32,
}

impl Simulator {
    pub fn from_config(config: Config) -> Self {
        fastrand::seed(0);

        let gates: Vec<Gate> = config
            .gates
            .iter()
            .map(|c| Gate { vertice: c.vertice })
            .collect();

        let pedestrians: Vec<_> = config
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

        let neighbor_lists = vec![vec![]; pedestrians.len()];

        Simulator {
            config,
            gates,
            pedestrians,
            neighbor_lists,
            step: 0,
        }
    }
}

impl Simulator {
    pub fn tick(&mut self, runtime: RuntimeKind) {
        // self.spawn_pedestrians();

        unsafe {
            let simulator: &Simulator = &*(self as *mut Simulator);

            match runtime {
                RuntimeKind::Single => {
                    let mut time = Instant::now();

                    self.pedestrians
                        .iter_mut()
                        .for_each(|p| p.calc_force_from_goal());

                    let duration_goal = Instant::now() - time;
                    time = Instant::now();

                    self.pedestrians
                        .iter_mut()
                        .for_each(|p| p.calc_pedestrians_interaction(simulator));

                    let duration_pedestrians = Instant::now() - time;

                    println!(
                        "Goal: {:.4}s, Pedestrians: {:.4}s",
                        duration_goal.as_secs_f64(),
                        duration_pedestrians.as_secs_f64()
                    );

                    // self.pedestrians
                    //     .iter_mut()
                    //     .for_each(|p| p.determine_accel(&*(simulator as *mut Simulator)));
                }
                RuntimeKind::Multi => {
                    self.pedestrians
                        .par_iter_mut()
                        .for_each(|p| p.determine_accel(&simulator.pedestrians, None));
                }
                RuntimeKind::Neighbor => {
                    if self.step % 20 == 0 {
                        self.neighbor_lists = self
                            .pedestrians
                            .par_iter()
                            .map(|p| p.create_neighbor_list(&simulator.pedestrians))
                            .collect();
                    }
                    self.pedestrians
                        .par_iter_mut()
                        .enumerate()
                        .for_each(|(i, p)| {
                            p.determine_accel(
                                &simulator.pedestrians,
                                Some(&simulator.neighbor_lists[i]),
                            )
                        });
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

        self.step += 1;
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
