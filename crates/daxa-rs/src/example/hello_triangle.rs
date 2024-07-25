#![feature(inherent_associated_types)]

mod hello_triangle_shared;

use daxa_rs::device_selector::best_gpu;
use daxa_rs::{AttachmentLoadOp, AttachmentStoreOp, BinarySemaphore, BinarySemaphoreInfo, ClearColorValue, ClearValue, CommandRecorderInfo, CommandSubmitInfo, Device, DeviceInfo, Extent, FixedList, ImageLayout, ImageMipArraySlice, ImageViewInfo, ImageViewType, Instance, InstanceFlags, InstanceInfo, Offset, Optional, PipelineShaderStageCreateFlags, PipelineStageFlags, PresentInfo, RasterPipeline, RasterPipelineInfo, Rect, RenderAttachment, RenderAttachmentInfo, RenderPassBeginInfo, ShaderInfo, Swapchain, SwapchainInfo, Viewport};
use std::{env, fs, mem, time};
use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::mem::swap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use glslang::limits::ResourceLimits;
use glslang::{Compiler, CompilerOptions, GlslProfile, Program, Shader, ShaderInput, ShaderSource, ShaderStage, SourceLanguage, SpirvVersion, Target, VulkanVersion};
use glslang::error::GlslangError::ParseError;
use glslang::include::{IncludeCallback, IncludeResult, IncludeType};
use lazy_static::lazy_static;
use raw_window_handle::HasWindowHandle;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowAttributes, WindowId};

struct App {
    window: Option<Window>,
    daxa: Option<Daxa>
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        self.window = Some(event_loop.create_window(WindowAttributes::default()).unwrap());
        self.daxa = Some(daxa(self.window.as_ref().unwrap()));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, window_id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::RedrawRequested => {
                let window = self.window.as_mut().unwrap();
                self.daxa.as_mut().unwrap().draw(window.inner_size().width, window.inner_size().height);
            }
            _ => {}
        }
    }
}


fn main() {
    let mut app = App { window: None, daxa: None };
    let mut event_loop = EventLoop::new().unwrap();
    event_loop.run_app_on_demand(&mut app).unwrap();
}

use hello_triangle_shared::Push;

lazy_static! {
    pub static ref GLSLANG_FILE_INCLUDER: Arc<Mutex<GlslangFileIncluder>> = Arc::new(Mutex::new(GlslangFileIncluder::default()));
}

type Header = String;
type Contents = String;

#[derive(Default)]
pub struct GlslangFileIncluder {
    virtual_files: HashMap<Header, Contents>,
    current_observed_hotload_files: HashMap<Header, time::Instant>,
    include_directories: Vec<PathBuf>,
}

impl GlslangFileIncluder {
    const MAX_INCLUSION_DEPTH: usize = 100;
    fn include(ty: IncludeType, header_name: &str, includer_name: &str, inclusion_depth: usize) -> Option<IncludeResult> {
        if inclusion_depth > Self::MAX_INCLUSION_DEPTH {
            None?
        }

        let mut this = GLSLANG_FILE_INCLUDER.lock().unwrap();

        if this.virtual_files.contains_key(header_name) {
            return Self::process(header_name.to_string(), this.virtual_files[header_name].clone())
        }

        let Some(full_path) = Self::full_path_to_file(&this.include_directories, header_name) else {
            None?
        };

        let mut contents = Self::load_shader_source_from_file(&full_path);

        this.virtual_files.insert(header_name.to_string(), contents.clone());

        return Self::process(full_path.to_str().unwrap().to_string(), contents);
    }

    fn process(name: String, contents: String) -> Option<IncludeResult> {
        let contents = contents.replace("#pragma once", "");
        Some(IncludeResult { name: name.into(), data: contents })
    }

    fn full_path_to_file(include_directories: &[PathBuf], name: &str) -> Option<PathBuf> {
        include_directories.iter().map(|dir| {
            let mut potential_path = dir.clone();
            potential_path.push(name);
            potential_path
        }).map(|x| dbg!(x)).filter(|path| fs::metadata(&path).is_ok()).next()
    }

    fn load_shader_source_from_file(path: &Path) -> String {
        fs::read_to_string(path).unwrap()
    }
}

pub struct ShaderCode(String);

impl ShaderCode {
    fn add(self, line: &str) -> Self {
        let Self(lines) = self;
        let mut result = String::new();
        result += line;
        result += "\n";
        result += &lines;
        result += "\n";
        Self(result)
    }
    fn vertex(self) -> Self {
        self.add("#define DAXA_SHADER_STAGE DAXA_SHADER_STAGE_VERTEX")
    }
    fn fragment(self) -> Self {
        self.add("#define DAXA_SHADER_STAGE DAXA_SHADER_STAGE_FRAGMENT")
    }
    fn ext(self) -> Self {
        self.add("#extension GL_GOOGLE_include_directive : require")
    }
}

fn shaders() -> (Vec<u32>, Vec<u32>) {
    const SHADER_SOURCE: &str = include_str!("hello_triangle.glsl");

    let compiler = Compiler::acquire().expect("Failed to acquire compiler");
    let opts = CompilerOptions {
        source_language: SourceLanguage::GLSL,
        target: Target::Vulkan {
            version: VulkanVersion::Vulkan1_3,
            spirv_version: SpirvVersion::SPIRV1_6,
        },
        version_profile: Some((460, GlslProfile::None)),
    };

    let includer = Some(GlslangFileIncluder::include as IncludeCallback);

    // Compile vertex shader
    let vertex_shader_code = ShaderCode(SHADER_SOURCE.to_string()).vertex().ext();
    let vertex_shader_source = ShaderSource::try_from(vertex_shader_code.0).expect("Vertex shader source");
    let vertex_shader_input = ShaderInput::new(
        &vertex_shader_source,
        ShaderStage::Vertex,
        &opts,
        includer,
    ).unwrap();
    let vertex_shader = match Shader::new(&compiler, vertex_shader_input) {
        Ok(s) => s,
        Err(ParseError(s)) => panic!("{s}"),
        _ => unreachable!(),
    };

    // Compile fragment shader
    let fragment_shader_code = ShaderCode(SHADER_SOURCE.to_string()).fragment().ext();
    let fragment_shader_source = ShaderSource::try_from(fragment_shader_code.0).expect("Fragment shader source");
    let fragment_shader_input = ShaderInput::new(
        &fragment_shader_source,
        ShaderStage::Fragment,
        &opts,
        includer
    ).unwrap();
    let fragment_shader = match Shader::new(&compiler, fragment_shader_input) {
        Ok(s) => s,
        Err(ParseError(s)) => panic!("{s}"),
        _ => unreachable!(),
    };

    // Create shader program and link shaders
    let mut program = Program::new(&compiler);
    program.add_shader(&vertex_shader);

    // Compile shaders to bytecode
    let vertex_shader_bytecode = program.compile(ShaderStage::Vertex).expect("Vertex shader bytecode");

    let mut program = Program::new(&compiler);
    program.add_shader(&fragment_shader);
    let fragment_shader_bytecode = program.compile(ShaderStage::Fragment).expect("Fragment shader bytecode");

    (vertex_shader_bytecode, fragment_shader_bytecode)
}

fn daxa(window: &impl HasWindowHandle) -> Daxa {
    GLSLANG_FILE_INCLUDER.lock().unwrap().include_directories.push("/home/solmidnight/work/daxa/daxa-rs/lib/daxa/include/".into());
    GLSLANG_FILE_INCLUDER.lock().unwrap().include_directories.push("/home/solmidnight/work/daxa/daxa-rs/crates/daxa-rs/src/example/".into());

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
         }).into(),
        fragment_shader_info: Some(ShaderInfo {
            byte_code: fragment.as_ptr(),
            byte_code_size: fragment.len() as u32,
            create_flags: PipelineShaderStageCreateFlags::empty(),
            required_subgroup_size: None.into(),
            entry_point: b"main".into(),
        }).into(),
        color_attachments: [RenderAttachment { format: swapchain.get_format(), ..Default::default() }].into(),
        push_constant_size: mem::size_of::<Push>() as u32,
        name: b"example-pipeline".into()
        , ..Default::default()
    });

    let image_available_semaphore = device.create_binary_semaphore(BinarySemaphoreInfo {
        name: b"image-available-semaphore".into()
    });
    let render_finished_semaphore = device.create_binary_semaphore(BinarySemaphoreInfo {
        name: b"render-finished-semaphore".into()
    });

    Daxa { device, swapchain, raster_pipeline, image_available_semaphore, render_finished_semaphore }
}

pub struct Daxa {
    device: Device,
    swapchain: Swapchain,
    raster_pipeline: RasterPipeline,
    pub render_finished_semaphore: BinarySemaphore,
    pub image_available_semaphore: BinarySemaphore,
}

impl Daxa {
    pub(crate) fn draw(&self, width: u32, height: u32) {
        let Self { device, swapchain, raster_pipeline, render_finished_semaphore, image_available_semaphore } = self;

        let mut command_recorder = device.create_command_recorder(CommandRecorderInfo::default());

        let image = swapchain.acquire_next_image();

        let image_view = device.create_image_view(ImageViewInfo {
            ty: ImageViewType::Type2d,
            format: swapchain.get_format(),
            image,
            slice: ImageMipArraySlice::default(),
            name: b"present-image-view".into(),
        });

        let render_area = Rect { offset: Offset { array: [0; 2] }, extent: Extent { array: [width, height]} };

        let render_pass = command_recorder.begin_render_pass(RenderPassBeginInfo {
            color_attachments: [RenderAttachmentInfo {
                image_view,
                layout: ImageLayout::ColorAttachmentOptimal,
                load_op: AttachmentLoadOp::Clear,
                store_op: AttachmentStoreOp::Store,
                clear_value: Some(ClearValue { color: ClearColorValue { float32: [1., 0., 1., 1.] } }).into(),
                resolve: Default::default(),
            }].into(),
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
            wait_binary_semaphores: [swapchain.current_present_semaphore()].into()
        });
    }
}
