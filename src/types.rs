use nalgebra::{Vector2, Vector3};

pub type Vec2 = Vector2<f32>;
pub type Vec3 = Vector3<f32>;

#[repr(C)]
#[repr(align(8))]
#[derive(Debug, Default, Clone, Copy, PartialEq, PartialOrd)]
pub struct Float2 {
    pub x: f32,
    pub y: f32,
}

impl From<Vec2> for Float2 {
    fn from(v: Vec2) -> Self {
        Float2 { x: v.x, y: v.y }
    }
}

impl Float2 {
    pub fn new(x: f32, y: f32) -> Self {
        Float2 { x, y }
    }
}
