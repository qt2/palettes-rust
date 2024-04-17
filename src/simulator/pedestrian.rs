use crate::Vec2;

const SFM_MASS: f32 = 80.0;
const SFM_TAU: f32 = 0.5;
const SFM_A: f32 = 2.0e+3;
const SFM_B: f32 = 0.08;
const SFM_K: f32 = 1.2e+5;
const SFM_KAPPA: f32 = 2.5e+5;

#[derive(Debug)]
pub struct Pedestrian {
    // pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub accel: Vec2,
    // pub start_id: usize,
    pub start: Vec2,
    // pub goal_id: usize,
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
            // start_id: 0,
            start: Vec2::zeros(),
            // goal_id: 0,
            goal: Vec2::zeros(),
            max_velocity: 1.34,
            has_arrived_goal: false,
            // accel_factors: Vec::new(),
        }
    }
}

impl Pedestrian {
    pub fn determine_accel(&mut self) {}

    fn calc_force_from_goal(&mut self) {}

    fn calc_force_from_pedestrian(&mut self) {}

    fn calc_force_from_wall(&mut self) {}

    pub fn walk(&mut self) {
        self.position += self.velocity * 0.1;
    }
}
