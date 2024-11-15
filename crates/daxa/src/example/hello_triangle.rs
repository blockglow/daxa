#![feature(inherent_associated_types)]

mod hello_triangle_shared;

use daxa::device_selector::best_gpu;
use daxa::{
    AttachmentLoadOp, AttachmentStoreOp, BinarySemaphore, BinarySemaphoreInfo, ClearColorValue,
    ClearValue, CommandRecorderInfo, CommandSubmitInfo, Device, DeviceInfo, Extent, FixedList,
    ImageLayout, ImageMipArraySlice, ImageViewInfo, ImageViewType, Instance, InstanceFlags,
    InstanceInfo, Offset, Optional, PipelineShaderStageCreateFlags, PipelineStageFlags,
    PresentInfo, RasterPipeline, RasterPipelineInfo, Rect, RenderAttachment, RenderAttachmentInfo,
    RenderPassBeginInfo, ShaderInfo, Swapchain, SwapchainInfo, Viewport,
};
use glslang::error::GlslangError::ParseError;
use glslang::include::{IncludeCallback, IncludeResult, IncludeType};
use glslang::limits::ResourceLimits;
use glslang::{
    Compiler, CompilerOptions, GlslProfile, Program, Shader, ShaderInput, ShaderSource,
    ShaderStage, SourceLanguage, SpirvVersion, Target, VulkanVersion,
};
use lazy_static::lazy_static;
use raw_window_handle::HasWindowHandle;
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::swap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use std::{env, fs, mem, time};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowAttributes, WindowId};
use daxa::{AttachmentLoadOp, AttachmentStoreOp, BinarySemaphoreInfo, ClearColorValue, ClearValue, CommandRecorderInfo, CommandSubmitInfo, Device, DeviceInfo, Extent, ImageLayout, ImageMipArraySlice, ImageViewInfo, ImageViewType, Instance, InstanceInfo, Offset, PipelineShaderStageCreateFlags, PipelineStageFlags, PresentInfo, RasterPipeline, RasterPipelineInfo, Rect, RenderAttachment, RenderAttachmentInfo, RenderPassBeginInfo, ShaderInfo, Swapchain, SwapchainInfo, Viewport};

struct App {
    window: Option<Window>,
    daxa: Option<Daxa>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(
            event_loop
                .create_window(WindowAttributes::default())
                .unwrap(),
        );
        self.daxa = Some(daxa(self.window.as_ref().unwrap()));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::RedrawRequested => {
                let window = self.window.as_mut().unwrap();
                self.daxa
                    .as_mut()
                    .unwrap()
                    .draw(window.inner_size().width, window.inner_size().height);
            }
            _ => {}
        }
    }
}

fn main() {
    let mut app = App {
        window: None,
        daxa: None,
    };
    let mut event_loop = EventLoop::new().unwrap();
    event_loop.run_app_on_demand(&mut app).unwrap();
}

use hello_triangle_shared::Push;

fn daxa(window: &impl HasWindowHandle) -> Daxa {
    GLSLANG_FILE_INCLUDER
        .lock()
        .unwrap()
        .include_directories
        .push("/home/solmidnight/work/daxa/daxa/crates/daxa/src/example/".into());

    let instance = Instance::new(InstanceInfo {
        app_name: b"hello-triangle-example".into(),
        ..InstanceInfo::default()
    });

    let (vertex, fragment) = shaders();

    let device = instance.create_device(DeviceInfo::default());

    let swapchain = device.create_swapchain(window, SwapchainInfo::default());

    let raster_pipeline = device.create_raster_pipeline(RasterPipelineInfo {
        vertex_shader_info: Some(ShaderInfo {
            byte_code: vertex.as_ptr(),
            byte_code_size: vertex.len() as u32,
            create_flags: PipelineShaderStageCreateFlags::empty(),
            required_subgroup_size: None.into(),
            entry_point: b"main".into(),
        })
        .into(),
        fragment_shader_info: Some(ShaderInfo {
            byte_code: fragment.as_ptr(),
            byte_code_size: fragment.len() as u32,
            create_flags: PipelineShaderStageCreateFlags::empty(),
            required_subgroup_size: None.into(),
            entry_point: b"main".into(),
        })
        .into(),
        color_attachments: [RenderAttachment {
            format: swapchain.get_format(),
            ..Default::default()
        }]
        .into(),
        push_constant_size: mem::size_of::<Push>() as u32,
        name: b"example-pipeline".into(),
        ..Default::default()
    });

    let image_available_semaphore = device.create_binary_semaphore(BinarySemaphoreInfo {
        name: b"image-available-semaphore".into(),
    });
    let render_finished_semaphore = device.create_binary_semaphore(BinarySemaphoreInfo {
        name: b"render-finished-semaphore".into(),
    });

    Daxa {
        device,
        swapchain,
        raster_pipeline,
    }
}

pub struct Daxa {
    device: Device,
    swapchain: Swapchain,
    raster_pipeline: RasterPipeline,
}

impl Daxa {
    pub(crate) fn draw(&self, width: u32, height: u32) {
        let Self {
            device,
            swapchain,
            raster_pipeline,
        } = self;

        let mut command_recorder = device.create_command_recorder(CommandRecorderInfo::default());

        let image = swapchain.acquire_next_image();

        let image_view = device.create_image_view(ImageViewInfo {
            ty: ImageViewType::Type2d,
            format: swapchain.get_format(),
            image,
            slice: ImageMipArraySlice::default(),
            name: b"present-image-view".into(),
        });

        let render_area = Rect {
            offset: Offset { array: [0; 2] },
            extent: Extent {
                array: [width, height],
            },
        };

        let render_pass = command_recorder.begin_render_pass(RenderPassBeginInfo {
            color_attachments: [RenderAttachmentInfo {
                image_view,
                layout: ImageLayout::ColorAttachmentOptimal,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: Some(ClearValue {
                    color: ClearColorValue {
                        float32: [1., 0., 1., 1.],
                    },
                })
                .into(),
                resolve: Default::default(),
            }]
            .into(),
            depth_attachment: None.into(),
            stencil_attachment: None.into(),
            render_area,
        });

        render_pass.bind_raster_pipeline(&raster_pipeline);
        render_pass.set_viewport(&Viewport {
            x: 0.0,
            y: 0.0,
            width: width as _,
            height: height as _,
            min_depth: 0.0,
            max_depth: 1.0,
        });
        render_pass.set_scissor(&render_area);
        render_pass.draw(0..3, 0..1);
        render_pass.end();

        let exe = command_recorder.complete();

        device.submit(&CommandSubmitInfo {
            wait_stages: PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            command_lists: [exe].into(),
            wait_binary_semaphores: [swapchain.current_acquire_semaphore()].into(),
            signal_binary_semaphores: [swapchain.current_present_semaphore()].into(),
            ..Default::default()
        });
        device.present(&PresentInfo {
            swapchain: *swapchain,
            wait_binary_semaphores: [swapchain.current_present_semaphore()].into(),
        });
    }
}
