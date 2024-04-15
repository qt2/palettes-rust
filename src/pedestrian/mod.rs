mod gate;
mod pedestrian;
mod wall;

use bevy::prelude::*;

pub struct PedestrianPlugin;

impl Plugin for PedestrianPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(FixedUpdate, tick);
    }
}

fn tick() {}
