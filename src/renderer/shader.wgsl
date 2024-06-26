struct Camera {
    position: vec2<f32>,
    rect: vec2<f32>,
    scale: f32,    
}
@group(0) @binding(0)
var<uniform> camera: Camera;

struct VertexInput {
    @location(0) position: vec2<f32>,
}

struct InstanceInput {
    @location(3) position: vec2<f32>,
    @location(4) color: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: u32,
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    var out: VertexOutput;
    let position = model.position + instance.position;
    let relative_position = camera.scale * (position - camera.position) / camera.rect;
    out.clip_position = vec4<f32>(relative_position, 0.0, 1.0);
    out.color = instance.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let color = unpack4x8unorm(in.color);
    return color;
}