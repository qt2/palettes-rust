use crate::Vec2;

use super::Simulator;

const SFM_MASS: f32 = 80.0;
const SFM_TAU: f32 = 0.5;
const SFM_A: f32 = 2.0e+3;
const SFM_B: f32 = 0.08;
const SFM_K: f32 = 1.2e+5;
const SFM_KAPPA: f32 = 2.5e+5;
const AGENT_SIZE: f32 = 0.4;

#[derive(Debug)]
pub struct Pedestrian {
    // pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub accel: Vec2,
    pub trip_id: usize,
    pub start_id: usize,
    // pub start: Vec2,
    pub goal_id: usize,
    pub goal: Vec2,
    pub max_velocity: f32,
    pub has_arrived_goal: bool,
    // pub accel_factors: Vec<Vec2>,
}

impl Default for Pedestrian {
    fn default() -> Self {
        Pedestrian {
            // id: 0,
            position: Vec2::zeros(),
            velocity: Vec2::zeros(),
            accel: Vec2::zeros(),
            trip_id: 0,
            start_id: 0,
            // start: Vec2::zeros(),
            goal_id: 0,
            goal: Vec2::zeros(),
            max_velocity: 1.34,
            has_arrived_goal: false,
            // accel_factors: Vec::new(),
        }
    }
}

impl Pedestrian {
    pub fn determine_accel(&mut self, simulator: &Simulator) {
        self.accel = Vec2::zeros();

        // calculate force from the goal
        let direction = (self.goal - self.position).normalize();
        self.accel += (self.max_velocity * direction - self.velocity) / SFM_TAU;

        // calculate force from pedestrians
        // skips if it is very close to its goal for easing congenstion.
        if (self.goal - self.position).magnitude_squared() > 4.0 {
            for pedestrian in &simulator.pedestrians {
                let direction = self.position - pedestrian.position;
                let d = direction.magnitude();
                if d > 2.0 || d < 1e-9 {
                    continue;
                }

                let r = 2.0 * AGENT_SIZE;
                let diff = r - d;
                let n = direction.normalize();

                let mut sf = Vec2::zeros();
                if diff >= 0.0 {
                    sf += n * SFM_K * diff;
                    let t = Vec2::new(-n.y, n.x);
                    let tvd = (pedestrian.velocity - self.velocity).dot(&t);
                    sf += t * (SFM_KAPPA * diff * tvd);
                }

                self.accel += sf / SFM_MASS;
            }
        }
    }

    pub fn walk(&mut self) {
        let previous_velocity = self.velocity;
        self.velocity += self.accel * 0.1;
        if self.velocity.magnitude() > self.max_velocity {
            self.velocity = self.velocity.normalize() * self.max_velocity;
        }

        self.position += (self.velocity + previous_velocity) * 0.5 * 0.1;

        if (self.position - self.goal).magnitude_squared() < 1.5 * 1.5 {
            self.has_arrived_goal = true;
        }
    }
}

#[repr(C)]
pub struct PedestriansT {
    pub x: *mut libc::c_float,
    pub y: *mut libc::c_float,
    pub vx: *mut libc::c_float,
    pub vy: *mut libc::c_float,
}

impl PedestriansT {
    pub fn from_pedestrians(pedestrians: &[Pedestrian]) -> Self {
        let mut x: Vec<_> = pedestrians.iter().map(|p| p.position.x).collect();
        let mut y: Vec<_> = pedestrians.iter().map(|p| p.position.y).collect();
        let mut vx: Vec<_> = pedestrians.iter().map(|p| p.velocity.x).collect();
        let mut vy: Vec<_> = pedestrians.iter().map(|p| p.velocity.y).collect();

        let ps = PedestriansT {
            x: x.as_mut_ptr(),
            y: y.as_mut_ptr(),
            vx: vx.as_mut_ptr(),
            vy: vy.as_mut_ptr(),
        };
        ps
    }
}
