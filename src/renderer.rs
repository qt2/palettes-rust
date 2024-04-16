use std::mem;

use eframe::{
    egui,
    wgpu::{
        self, include_wgsl,
        util::{BufferInitDescriptor, DeviceExt},
        Buffer, BufferAddress, BufferUsages, FragmentState, MultisampleState,
        PipelineLayoutDescriptor, PrimitiveState, RenderPipeline, RenderPipelineDescriptor,
        VertexAttribute, VertexBufferLayout, VertexState,
    },
};

pub struct Renderer {}

impl Renderer {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let render_state = cc.wgpu_render_state.as_ref().unwrap();
        let device = &render_state.device;
        let shader = device.create_shader_module(include_wgsl!("./shader.wgsl"));
        // let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        //     label: Some("pedestrians"),
        //     entries: &[],
        // });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("pedestrians"),
            // bind_group_layouts: &[&bind_group_layout],
            bind_group_layouts: &[],
            ..Default::default()
        });
        let pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("pedestrians"),
            layout: Some(&pipeline_layout),
            vertex: VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
            },
            fragment: Some(FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(render_state.target_format.into())],
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview: None,
        });
        let vertex_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: Some("vertex buffer"),
            contents: bytemuck::cast_slice(&[
                Vertex {
                    position: [0.0, 1.0],
                },
                Vertex {
                    position: [-0.5, -0.5],
                },
                Vertex {
                    position: [0.5, -0.5],
                },
            ]),
            usage: BufferUsages::VERTEX,
        });

        render_state
            .renderer
            .write()
            .callback_resources
            .insert(PedestrianRenderResources {
                pipeline,
                vertex_buffer,
                // bind_group,
            });

        Renderer {}
    }

    pub fn draw_canvas(&mut self, ui: &mut egui::Ui) {
        let (rect, response) =
            ui.allocate_exact_size(egui::Vec2::splat(300.0), egui::Sense::drag());
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            CustomCallback {},
        ));
    }
}

struct CustomCallback {}

impl egui_wgpu::CallbackTrait for CustomCallback {
    fn paint<'a>(
        &'a self,
        info: egui::PaintCallbackInfo,
        render_pass: &mut eframe::wgpu::RenderPass<'a>,
        callback_resources: &'a egui_wgpu::CallbackResources,
    ) {
        let resources: &PedestrianRenderResources = callback_resources.get().unwrap();

        render_pass.set_pipeline(&resources.pipeline);
        render_pass.set_vertex_buffer(0, resources.vertex_buffer.slice(..));
        render_pass.draw(0..3, 0..1);
    }
}

struct PedestrianRenderResources {
    pipeline: RenderPipeline,
    vertex_buffer: Buffer,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 2],
}

impl Vertex {
    const ATTRIBS: [VertexAttribute; 1] = wgpu::vertex_attr_array![0  => Float32x2];

    fn desc() -> VertexBufferLayout<'static> {
        VertexBufferLayout {
            array_stride: mem::size_of::<Self>() as BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}
