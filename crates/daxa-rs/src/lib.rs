#![feature(maybe_uninit_uninit_array)]

use crate::device_selector::best_gpu;
use bitflags::bitflags;
use derive_more::{Deref, DerefMut, Into};
use std::convert::identity;
use std::ffi::c_char;
use std::{ffi, fmt, mem, ptr};
use std::mem::{MaybeUninit, swap};
use std::ops::Range;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::AtomicPtr;
use raw_window_handle::{HasRawWindowHandle, HasWindowHandle, RawWindowHandle, WindowHandle};
use uuid::Uuid;
use daxa_sys::{daxa_BinarySemaphore, daxa_Bool8, daxa_Device, daxa_ExecutableCommandList, daxa_NativeWindowHandle, daxa_TimelinePair, VkCompareOp, VkFormat, VkPipelineStageFlags};
use crate::surface_format_selector::default_format_score;

pub type Flags = u64;

bitflags! {
    pub struct InstanceFlags: Flags {
        const DEBUG_UTIL = 0b00000001;
        const PARENT_MUST_OUTLIVE_CHILD = 0b00000010;
    }
}

pub struct InstanceInfo {
    pub flags: InstanceFlags,
    pub app_name: SmallString,
    pub engine_name: SmallString,
}

impl Default for InstanceInfo {
    fn default() -> Self {
        Self {
            flags: InstanceFlags::empty(),
            app_name: b"default-daxa-app".into(),
            engine_name: b"default-daxa-engine".into(),
        }
    }
}

pub const SMALL_STRING_CAPACITY: usize = 63;

#[repr(C)]
#[derive(Deref, DerefMut, Into)]
pub struct SmallString(daxa_sys::daxa_SmallString);

impl<T: AsRef<[u8]>> From<T> for SmallString {
    fn from(value: T) -> Self {
        let mut string = SmallString(daxa_sys::daxa_SmallString {
            data: [0; SMALL_STRING_CAPACITY],
            size: 0,
        });
        //This is a copy Patrick, not a clone. ;p
        value
            .as_ref()
            .iter()
            .copied()
            .enumerate()
            .for_each(|(i, c)| {
                debug_assert!(i <= SMALL_STRING_CAPACITY, "small string is too long");
                string.data[i] = c as _;
                string.size = (i + 1) as _;
            });
        string
    }
}

pub struct Instance(daxa_sys::daxa_Instance);

impl Instance {
    pub fn new(info: InstanceInfo) -> Self {
        let InstanceInfo {
            flags,
            app_name,
            engine_name,
        } = info;
        let info = daxa_sys::daxa_InstanceInfo {
            flags: flags.bits(),
            engine_name: engine_name.into(),
            app_name: app_name.into(),
        };
        let mut instance = MaybeUninit::uninit();
        let instance = unsafe {
            daxa_sys::daxa_create_instance(&info, instance.as_mut_ptr());
            instance.assume_init()
        };
        Self(instance)
    }

    pub fn create_device(&self, info: impl IntoDeviceInfo) -> Device {
        let Self(instance) = self;
        let DeviceInfo { flags, max_allowed_buffers, max_allowed_acceleration_structures, max_allowed_images, max_allowed_samplers, selector, name } = info.into_device_info();
        let device = unsafe {
            let info = daxa_sys::daxa_DeviceInfo {
                flags: flags.bits(),
                max_allowed_buffers,
                max_allowed_images,
                max_allowed_samplers,
                max_allowed_acceleration_structures,
                selector: selector.map(|x| mem::transmute(x)),
                name: name.into(),
            };
            let mut device = MaybeUninit::uninit();
            let result = daxa_sys::daxa_instance_create_device(*instance, &info, device.as_mut_ptr());
            dbg!(result);
            device.assume_init()
        };
        Device(device)
    }
}

pub trait IntoDeviceInfo {
    fn into_device_info(self) -> DeviceInfo;
}

impl IntoDeviceInfo for DeviceInfo {
    fn into_device_info(self) -> DeviceInfo {
        self
    }
}

impl IntoDeviceInfo for DeviceSelector {
    fn into_device_info(self) -> DeviceInfo {
        DeviceInfo {
            selector: Some(self),
            ..Default::default()
        }
    }
}
pub type DeviceSelector = extern "C" fn(&DeviceProperties) -> DeviceScore;


pub mod device_selector {
    use crate::{DeviceProperties, DeviceScore, DeviceType};

    pub extern "C" fn best_gpu(device_props: &DeviceProperties) -> DeviceScore {
        let mut score = 0;
        score += match device_props.device_type {
            DeviceType::Integrated => 100,
            DeviceType::Discrete => 10000,
            _ => 0,
        };
        score += (device_props.limits.max_memory_allocation_count / 10) as i32;
        println!("score for {} is {score}", device_props.device_name);
        score
    }
}

bitflags! {
    pub struct DeviceFlags: u32 {}
}

pub type DeviceScore = i32;

pub struct DeviceInfo {
    flags: DeviceFlags,
    max_allowed_images: u32,
    max_allowed_buffers: u32,
    max_allowed_samplers: u32,
    max_allowed_acceleration_structures: u32,
    selector: Option<DeviceSelector>,
    name: SmallString,
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            flags: DeviceFlags::empty(),
            selector: Some(best_gpu),
            max_allowed_acceleration_structures: 10000,
            max_allowed_buffers: 10000,
            max_allowed_images: 10000,
            max_allowed_samplers: 10000,
            name: "device".into(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(i32)]
pub enum DeviceType {
    Other,
    Integrated,
    Discrete,
    Virtual,
    Cpu,
}

#[repr(C)]
#[derive(Deref, DerefMut, Clone, Copy)]
pub struct DeviceName([c_char; 256]);

impl fmt::Display for DeviceName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Create a CStr from the array of c_char
        let c_str = unsafe { ffi::CStr::from_ptr(self.0.as_ptr()) };

        // Convert CStr to a Rust str, handling potential errors
        match c_str.to_str() {
            Ok(s) => write!(f, "{}", s),
            Err(_) => write!(f, "<invalid UTF-8>"),
        }
    }
}

#[repr(C)]
pub struct Optional<T> {
    value: MaybeUninit<T>,
    has_value: bool,
}

impl<T: Copy> Clone for Optional<T> {
    fn clone(&self) -> Self {
         Self {
             value: self.value,
             has_value: self.has_value
         }
    }
}

impl<T> Default for Optional<T> {
    fn default() -> Self {
        None.into()
    }
}

impl<T> Optional<T> {
    fn as_option_ref(&self) -> Option<&T> {
    self.has_value.then(|| unsafe {   self.value.assume_init_ref()
    })
    }
    fn as_option_mut(&mut self) -> Option<&mut T> {
       self.has_value.then(||  unsafe {   self.value.assume_init_mut()
        })
    }
}

impl<T> From<Optional<T>> for Option<T> {
    fn from(value: Optional<T>) -> Self {
        value.has_value.then(|| unsafe { value.value.assume_init() })
    }
}

impl<T> From<Option<T>> for Optional<T> {
    fn from(value: Option<T>) -> Self {
        match value {
            Some(value) => Optional { value: MaybeUninit::new(value), has_value: true, },
            None => Optional { value: MaybeUninit::uninit(), has_value: false }
        }
    }
}

#[repr(C)]
pub struct DeviceProperties {
    pub vulkan_api_version: u32,
    pub driver_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub device_type: DeviceType,
    pub device_name: DeviceName,
    pub pipeline_cache_uuid: Uuid,
    pub limits: DeviceLimits,
    pub mesh_shader_properties: Optional<MeshShaderProperties>,
    pub ray_tracing_pipeline_properties: Optional<RayTracingPipelineProperties>,
    pub acceleration_structure_properties: Optional<AccelerationStructureProperties>,
    pub ray_tracing_invocation_reorder_properties: Optional<RayTracingInvocationReorderProperties>,
}

#[derive(Clone, Copy)]
pub struct DeviceLimits {
    pub max_image_dimension1d: u32,
    pub max_image_dimension2d: u32,
    pub max_image_dimension3d: u32,
    pub max_image_dimension_cube: u32,
    pub max_image_array_layers: u32,
    pub max_texel_buffer_elements: u32,
    pub max_uniform_buffer_range: u32,
    pub max_storage_buffer_range: u32,
    pub max_push_constants_size: u32,
    pub max_memory_allocation_count: u32,
    pub max_sampler_allocation_count: u32,
    pub buffer_image_granularity: u64,
    pub sparse_address_space_size: u64,
    pub max_bound_descriptor_sets: u32,
    pub max_per_stage_descriptor_samplers: u32,
    pub max_per_stage_descriptor_uniform_buffers: u32,
    pub max_per_stage_descriptor_storage_buffers: u32,
    pub max_per_stage_descriptor_sampled_images: u32,
    pub max_per_stage_descriptor_storage_images: u32,
    pub max_per_stage_descriptor_input_attachments: u32,
    pub max_per_stage_resources: u32,
    pub max_descriptor_set_samplers: u32,
    pub max_descriptor_set_uniform_buffers: u32,
    pub max_descriptor_set_uniform_buffers_dynamic: u32,
    pub max_descriptor_set_storage_buffers: u32,
    pub max_descriptor_set_storage_buffers_dynamic: u32,
    pub max_descriptor_set_sampled_images: u32,
    pub max_descriptor_set_storage_images: u32,
    pub max_descriptor_set_input_attachments: u32,
    pub max_vertex_input_attributes: u32,
    pub max_vertex_input_bindings: u32,
    pub max_vertex_input_attribute_offset: u32,
    pub max_vertex_input_binding_stride: u32,
    pub max_vertex_output_components: u32,
    pub max_tessellation_generation_level: u32,
    pub max_tessellation_patch_size: u32,
    pub max_tessellation_control_per_vertex_input_components: u32,
    pub max_tessellation_control_per_vertex_output_components: u32,
    pub max_tessellation_control_per_patch_output_components: u32,
    pub max_tessellation_control_total_output_components: u32,
    pub max_tessellation_evaluation_input_components: u32,
    pub max_tessellation_evaluation_output_components: u32,
    pub max_geometry_shader_invocations: u32,
    pub max_geometry_input_components: u32,
    pub max_geometry_output_components: u32,
    pub max_geometry_output_vertices: u32,
    pub max_geometry_total_output_components: u32,
    pub max_fragment_input_components: u32,
    pub max_fragment_output_attachments: u32,
    pub max_fragment_dual_src_attachments: u32,
    pub max_fragment_combined_output_resources: u32,
    pub max_compute_shared_memory_size: u32,
    pub max_compute_work_group_count: [u32; 3usize],
    pub max_compute_work_group_invocations: u32,
    pub max_compute_work_group_size: [u32; 3usize],
    pub sub_pixel_precision_bits: u32,
    pub sub_texel_precision_bits: u32,
    pub mipmap_precision_bits: u32,
    pub max_draw_indexed_index_value: u32,
    pub max_draw_indirect_count: u32,
    pub max_sampler_lod_bias: f32,
    pub max_sampler_anisotropy: f32,
    pub max_viewports: u32,
    pub max_viewport_dimensions: [u32; 2usize],
    pub viewport_bounds_range: [f32; 2usize],
    pub viewport_sub_pixel_bits: u32,
    pub min_memory_map_alignment: usize,
    pub min_texel_buffer_offset_alignment: u64,
    pub min_uniform_buffer_offset_alignment: u64,
    pub min_storage_buffer_offset_alignment: u64,
    pub min_texel_offset: i32,
    pub max_texel_offset: u32,
    pub min_texel_gather_offset: i32,
    pub max_texel_gather_offset: u32,
    pub min_interpolation_offset: f32,
    pub max_interpolation_offset: f32,
    pub sub_pixel_interpolation_offset_bits: u32,
    pub max_framebuffer_width: u32,
    pub max_framebuffer_height: u32,
    pub max_framebuffer_layers: u32,
    pub framebuffer_color_sample_counts: u32,
    pub framebuffer_depth_sample_counts: u32,
    pub framebuffer_stencil_sample_counts: u32,
    pub framebuffer_no_attachments_sample_counts: u32,
    pub max_color_attachments: u32,
    pub sampled_image_color_sample_counts: u32,
    pub sampled_image_integer_sample_counts: u32,
    pub sampled_image_depth_sample_counts: u32,
    pub sampled_image_stencil_sample_counts: u32,
    pub storage_image_sample_counts: u32,
    pub max_sample_mask_words: u32,
    pub timestamp_compute_and_graphics: u32,
    pub timestamp_period: f32,
    pub max_clip_distances: u32,
    pub max_cull_distances: u32,
    pub max_combined_clip_and_cull_distances: u32,
    pub discrete_queue_priorities: u32,
    pub point_size_range: [f32; 2usize],
    pub line_width_range: [f32; 2usize],
    pub point_size_granularity: f32,
    pub line_width_granularity: f32,
    pub strict_lines: u32,
    pub standard_sample_locations: u32,
    pub optimal_buffer_copy_offset_alignment: u64,
    pub optimal_buffer_copy_row_pitch_alignment: u64,
    pub non_coherent_atom_size: u64,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct MeshShaderProperties {
    pub max_task_work_group_total_count: u32,
    pub max_task_work_group_count: [u32; 3usize],
    pub max_task_work_group_invocations: u32,
    pub max_task_work_group_size: [u32; 3usize],
    pub max_task_payload_size: u32,
    pub max_task_shared_memory_size: u32,
    pub max_task_payload_and_shared_memory_size: u32,
    pub max_mesh_work_group_total_count: u32,
    pub max_mesh_work_group_count: [u32; 3usize],
    pub max_mesh_work_group_invocations: u32,
    pub max_mesh_work_group_size: [u32; 3usize],
    pub max_mesh_shared_memory_size: u32,
    pub max_mesh_payload_and_shared_memory_size: u32,
    pub max_mesh_output_memory_size: u32,
    pub max_mesh_payload_and_output_memory_size: u32,
    pub max_mesh_output_components: u32,
    pub max_mesh_output_vertices: u32,
    pub max_mesh_output_primitives: u32,
    pub max_mesh_output_layers: u32,
    pub max_mesh_multiview_view_count: u32,
    pub mesh_output_per_vertex_granularity: u32,
    pub mesh_output_per_primitive_granularity: u32,
    pub max_preferred_task_work_group_invocations: u32,
    pub max_preferred_mesh_work_group_invocations: u32,
    pub prefers_local_invocation_vertex_output: bool,
    pub prefers_local_invocation_primitive_output: bool,
    pub prefers_compact_vertex_output: bool,
    pub prefers_compact_primitive_output: bool,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct RayTracingPipelineProperties {
    pub shader_group_handle_size: u32,
    pub max_ray_recursion_depth: u32,
    pub max_shader_group_stride: u32,
    pub shader_group_base_alignment: u32,
    pub shader_group_handle_capture_replay_size: u32,
    pub max_ray_dispatch_invocation_count: u32,
    pub shader_group_handle_alignment: u32,
    pub max_ray_hit_attribute_size: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct AccelerationStructureProperties {
    pub max_geometry_count: u64,
    pub max_instance_count: u64,
    pub max_primitive_count: u64,
    pub max_per_stage_descriptor_acceleration_structures: u32,
    pub max_per_stage_descriptor_update_after_bind_acceleration_structures: u32,
    pub max_descriptor_set_acceleration_structures: u32,
    pub max_descriptor_set_update_after_bind_acceleration_structures: u32,
    pub min_acceleration_structure_scratch_offset_alignment: u32,
}
#[derive(Clone, Copy)]
#[repr(C)]
pub struct RayTracingInvocationReorderProperties {
    pub invocation_reorder_mode: u32,
}

bitflags! {
    #[derive(Clone, Copy)]
    pub struct ResolveModeFlag: u32 {
        const NONE = 0;
        const SAMPLE_ZERO = 0x00000001;
        const AVERAGE = 0x00000002;
        const MIN = 0x00000004;
        const MAX = 0x00000008;
        // Provided by VK_ANDROID_external_format_resolve with VK_KHR_dynamic_rendering or VK_VERSION_1_3
        const EXTERNAL_FORMAT_DOWNSAMPLE_ANDROID = 0x00000010;
        // Provided by VK_KHR_depth_stencil_resolve
        const NONE_KHR = Self::NONE.bits();
        // Provided by VK_KHR_depth_stencil_resolve
        const SAMPLE_ZERO_KHR = Self::SAMPLE_ZERO.bits();
        // Provided by VK_KHR_depth_stencil_resolve
        const AVERAGE_KHR = Self::AVERAGE.bits();
        // Provided by VK_KHR_depth_stencil_resolve
        const MIN_KHR = Self::MIN.bits();
        // Provided by VK_KHR_depth_stencil_resolve
        const MAX_KHR = Self::MAX.bits();
    }
}


pub struct Device(daxa_sys::daxa_Device);

pub struct CommandRecorder(daxa_sys::daxa_CommandRecorder);
#[derive(Clone, Copy)]
pub struct ExecutableCommandList(daxa_sys::daxa_ExecutableCommandList);

pub struct RenderPass<'a>(&'a mut CommandRecorder);

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Viewport {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
    pub min_depth: f32,
    pub max_depth: f32,
}

impl<'a> RenderPass<'a> {
    pub fn set_viewport(&self, viewport: &Viewport) {
        unsafe { daxa_sys::daxa_cmd_set_viewport(self.0.0, viewport as *const _ as *const _); }
    }
    pub fn set_scissor(&self, scissor: &Rect) {
        unsafe { daxa_sys::daxa_cmd_set_scissor(self.0.0, scissor as *const _ as *const _); }
    }
    pub fn bind_raster_pipeline(&self, raster_pipeline: &RasterPipeline) {
        unsafe { daxa_sys::daxa_cmd_set_raster_pipeline(self.0.0, raster_pipeline.0); }
    }
    pub fn draw(&self, indices: Range<u32>, instances: Range<u32>) {
        unsafe { daxa_sys::daxa_cmd_draw(self.0.0, &daxa_sys::daxa_DrawInfo { vertex_count: indices.end - indices.start, instance_count: instances.end - instances.start, first_vertex: indices.start, first_instance: instances.start }) };
    }
    pub fn end(self) {
        unsafe { daxa_sys::daxa_cmd_end_renderpass(self.0.0) }
    }
}

impl CommandRecorder {
    pub fn begin_render_pass(&mut self, info: RenderPassBeginInfo) -> RenderPass {
        let Self(cmd) = self;
        unsafe { daxa_sys::daxa_cmd_begin_renderpass(*cmd, &info as *const _ as *const _) };
        RenderPass(self)
    }
    pub fn complete(self) -> ExecutableCommandList {
        let Self(recorder) = self;
        let exe = unsafe {
            let mut exe = MaybeUninit::uninit();
            daxa_sys::daxa_cmd_complete_current_commands(recorder, exe.as_mut_ptr());
            exe.assume_init()
        };
        ExecutableCommandList(exe)
    }
}

#[repr(C)]
pub struct RenderPassBeginInfo {
    pub color_attachments: FixedList<RenderAttachmentInfo, 8>,
    pub depth_attachment: Optional<RenderAttachmentInfo>,
    pub stencil_attachment: Optional<RenderAttachmentInfo>,
    pub render_area: Rect
}

#[derive(Clone, Copy)]
pub struct Rect {
    pub offset: Offset<2>,
    pub extent: Extent<2>,
}

#[derive(Clone, Copy)]
pub struct Offset<const D: usize> {
    pub array: [i32; D],
}

#[derive(Clone, Copy)]
pub struct Extent<const D: usize> {
    pub array: [u32; D],
}

// pub struct RenderPass(daxa_sys::daxa_Rend);

#[repr(C)]
pub struct CommandRecorderInfo {
    name: SmallString
}

impl Default for CommandRecorderInfo {
    fn default() -> Self {
        Self {
            name: b"command-recorder".into(),
        }
    }
}

#[repr(C)]
pub struct ImageViewInfo {
    pub ty: ImageViewType,
    pub format: Format,
    pub image: ImageId,
    pub slice: ImageMipArraySlice,
    pub name: SmallString,
}
#[repr(C)]
pub struct ImageMipArraySlice {
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

impl Default for ImageMipArraySlice {
    fn default() -> Self {
        Self {
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        }
    }
}
#[repr(u32)]
pub enum ImageViewType {
    Type1d = 0,
    Type2d = 1,
    Type3d = 2,
    TypeCube = 3,
    Type1dArray = 4,
    Type2dArray = 5,
    TypeCubeArray = 6,
}

bitflags! {

    pub struct PipelineStageFlags: u64 {
        const NONE = 0;
        const TOP_OF_PIPE = 0x00000001;
        const DRAW_INDIRECT = 0x00000002;
        const VERTEX_INPUT = 0x00000004;
        const VERTEX_SHADER = 0x00000008;
        const TESSELLATION_CONTROL_SHADER = 0x00000010;
        const TESSELLATION_EVALUATION_SHADER = 0x00000020;
        const GEOMETRY_SHADER = 0x00000040;
        const FRAGMENT_SHADER = 0x00000080;
        const EARLY_FRAGMENT_TESTS = 0x00000100;
        const LATE_FRAGMENT_TESTS = 0x00000200;
        const COLOR_ATTACHMENT_OUTPUT = 0x00000400;
        const COMPUTE_SHADER = 0x00000800;
        const ALL_TRANSFER = 0x00001000;
        const TRANSFER = 0x00001000;
        const BOTTOM_OF_PIPE = 0x00002000;
        const HOST = 0x00004000;
        const ALL_GRAPHICS = 0x00008000;
        const ALL_COMMANDS = 0x00010000;
        const COPY = 0x100000000;
        const RESOLVE = 0x200000000;
        const BLIT = 0x400000000;
        const CLEAR = 0x800000000;
        const INDEX_INPUT = 0x1000000000;
        const VERTEX_ATTRIBUTE_INPUT = 0x2000000000;
        const PRE_RASTERIZATION_SHADERS = 0x4000000000;
        const VIDEO_DECODE_KHR = 0x04000000;
        const VIDEO_ENCODE_KHR = 0x08000000;
        const TRANSFORM_FEEDBACK_EXT = 0x01000000;
        const CONDITIONAL_RENDERING_EXT = 0x00040000;
        const COMMAND_PREPROCESS_NV = 0x00020000;
        const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR = 0x00400000;
        const SHADING_RATE_IMAGE_NV = 0x00400000;
        const ACCELERATION_STRUCTURE_BUILD_KHR = 0x02000000;
        const RAY_TRACING_SHADER_KHR = 0x00200000;
        const RAY_TRACING_SHADER_NV = 0x00200000;
        const ACCELERATION_STRUCTURE_BUILD_NV = 0x02000000;
        const FRAGMENT_DENSITY_PROCESS_EXT = 0x00800000;
        const TASK_SHADER_NV = 0x00080000;
        const MESH_SHADER_NV = 0x00100000;
        const TASK_SHADER_EXT = 0x00080000;
        const MESH_SHADER_EXT = 0x00100000;
        const SUBPASS_SHADER_HUAWEI = 0x8000000000;
        const SUBPASS_SHADING_HUAWEI = 0x8000000000;
        const INVOCATION_MASK_HUAWEI = 0x10000000000;
        const ACCELERATION_STRUCTURE_COPY_KHR = 0x10000000;
        const MICROMAP_BUILD_EXT = 0x40000000;
        const CLUSTER_CULLING_SHADER_HUAWEI = 0x20000000000;
        const OPTICAL_FLOW_NV = 0x20000000;
    }
}

#[derive(Clone, Copy)]
pub struct BinarySemaphore(daxa_sys::daxa_BinarySemaphore);
#[derive(Clone, Copy)]
pub struct TimelinePair(daxa_sys::daxa_TimelinePair);

#[repr(C)]
pub struct CommandSubmitInfo {
    pub wait_stages: PipelineStageFlags,
    pub command_lists: FixedList<ExecutableCommandList, 8>,
    pub wait_binary_semaphores: FixedList<BinarySemaphore, 8>,
    pub signal_binary_semaphores: FixedList<BinarySemaphore, 8>,
    pub wait_timeline_semaphores: FixedList<TimelinePair, 8>,
    pub signal_timeline_semaphores: FixedList<TimelinePair, 8>,
}

impl Default for CommandSubmitInfo{
    fn default() -> Self {
        Self {
            wait_stages: PipelineStageFlags::NONE,
            command_lists: Default::default(),
            wait_timeline_semaphores: Default::default(),
            wait_binary_semaphores: Default::default(),
            signal_timeline_semaphores: Default::default(),
            signal_binary_semaphores: Default::default()
        }
    }
}

#[repr(C)]
pub struct PresentInfo {
    pub swapchain: Swapchain,
    pub wait_binary_semaphores: FixedList<BinarySemaphore, 8>,
}
pub struct BinarySemaphoreInfo {
    pub name: SmallString
}

impl Device {
    pub fn create_binary_semaphore(&self, info: BinarySemaphoreInfo) -> BinarySemaphore {
        let Self(device) = self;
        let semaphore = unsafe {
            let mut semaphore = MaybeUninit::uninit();
            daxa_sys::daxa_dvc_create_binary_semaphore(*device, &daxa_sys::daxa_BinarySemaphoreInfo {
                name: info.name.into()
            }, semaphore.as_mut_ptr());
            semaphore.assume_init()
        };
        BinarySemaphore(semaphore)
    }
    pub fn present(&self, info: &PresentInfo) {
        let Self(device) = self;
        unsafe { daxa_sys::daxa_dvc_present(*device, &daxa_sys::daxa_PresentInfo {
            wait_binary_semaphores: info.wait_binary_semaphores.data.as_ptr() as *const _,
            wait_binary_semaphore_count: info.wait_binary_semaphores.len as _,
            swapchain: info.swapchain.0,
        });}
    }
    pub fn submit(&self, info: &CommandSubmitInfo) {
        let Self(device) = self;
        unsafe { daxa_sys::daxa_dvc_submit(*device, &daxa_sys::daxa_CommandSubmitInfo {
            wait_stages: 0,
            command_lists: info.command_lists.data.as_ptr() as *const _,
            command_list_count: info.command_lists.len as _,
            wait_binary_semaphores: info.wait_binary_semaphores.data.as_ptr() as *const _,
            wait_binary_semaphore_count: info.wait_binary_semaphores.len as _,
            signal_binary_semaphores: info.signal_binary_semaphores.data.as_ptr() as *const _,
            signal_binary_semaphore_count: info.signal_binary_semaphores.len as _,
            wait_timeline_semaphores: info.wait_timeline_semaphores.data.as_ptr() as *const _,
            wait_timeline_semaphore_count: info.wait_timeline_semaphores.len as _ ,
            signal_timeline_semaphores: info.signal_timeline_semaphores.data.as_ptr() as *const _,
            signal_timeline_semaphore_count: info.signal_timeline_semaphores.len as _,
        });}
    }
    pub fn create_image_view(&self, info: ImageViewInfo) -> ImageViewId {
        let Self(device) = self;
        let image_view_id = unsafe {
            let mut image_view_id = MaybeUninit::uninit();
            daxa_sys::daxa_dvc_create_image_view(*device, &info as *const  _ as *const _, image_view_id.as_mut_ptr());
            image_view_id.assume_init()
        };
        ImageViewId(image_view_id)
    }
    pub fn create_command_recorder(&self, info: CommandRecorderInfo) -> CommandRecorder {
        let Self(device) = self;
        let command_recorder = unsafe {
            let mut command_recorder = MaybeUninit::uninit();
            daxa_sys::daxa_dvc_create_command_recorder(*device, &info as *const  _ as *const _, command_recorder.as_mut_ptr());
            command_recorder.assume_init()
        };
        CommandRecorder(command_recorder)
    }
    pub fn create_raster_pipeline(&self, info: RasterPipelineInfo) -> RasterPipeline {
        let Self(device) = self;
        let raster_pipeline = unsafe {
            let mut pipeline = MaybeUninit::uninit();
            daxa_sys::daxa_dvc_create_raster_pipeline(*device, &info as *const _ as *const _, pipeline.as_mut_ptr());
            pipeline.assume_init()
        };
        RasterPipeline(raster_pipeline)
    }
    pub fn create_swapchain(&self, window: &impl HasWindowHandle, info: SwapchainInfo) -> Swapchain {
        let Self(device) = self;
        let SwapchainInfo { surface_format_selector, present_operation, present_mode, image_usage, name} = info;
        let swapchain = unsafe {
            let info = daxa_sys::daxa_SwapchainInfo {
                surface_format_selector: mem::transmute(surface_format_selector),
                present_mode: present_mode as _,
                present_operation: present_operation.bits(),
                image_usage: image_usage.bits(),
                max_allowed_frames_in_flight: 3,
                name: name.into(),
                ..mem::zeroed()
            };

            let info = match window.raw_window_handle().unwrap() {
                    RawWindowHandle::Xlib(mut handle) =>
                    daxa_sys::daxa_SwapchainInfo {
                        native_window: handle.window as *mut _,
                        native_window_platform: daxa_sys::daxa_NativeWindowPlatform_DAXA_NATIVE_WINDOW_PLATFORM_XLIB_API,
                        ..info
                    },
                RawWindowHandle::Wayland(mut handle) =>
                    daxa_sys::daxa_SwapchainInfo {
                        native_window: handle.surface.as_ptr(),
                        native_window_platform: daxa_sys::daxa_NativeWindowPlatform_DAXA_NATIVE_WINDOW_PLATFORM_WAYLAND_API,
                        ..info
                    },
                _ => todo!()
            };
            let mut swapchain = MaybeUninit::uninit();
            let result = daxa_sys::daxa_dvc_create_swapchain(*device, &info, swapchain.as_mut_ptr());
            dbg!(result);
            swapchain.assume_init()
        };
        Swapchain(swapchain)
    }
}

bitflags! {
    pub struct SurfaceTransform: u32 {
        const IDENTITY = 0x00000001;
        const ROTATE_90 = 0x00000002;
        const ROTATE_180 = 0x00000004;
        const ROTATE_270 = 0x00000008;
        const HORIZONTAL_MIRROR = 0x00000010;
        const HORIZONTAL_MIRROR_ROTATE_90 = 0x00000020;
        const HORIZONTAL_MIRROR_ROTATE_180 = 0x00000040;
        const HORIZONTAL_MIRROR_ROTATE_270 = 0x00000080;
        const INHERIT = 0x00000100;
    }
}

pub type SurfaceFormatScore = i32;
pub type SurfaceFormatSelector = extern "C" fn(Format) -> SurfaceFormatScore;

pub struct SwapchainInfo {
    surface_format_selector:  SurfaceFormatSelector,
    present_mode: PresentMode,
    present_operation: SurfaceTransform,
    image_usage: ImageUsage,
    name: SmallString,
}
impl Default for SwapchainInfo {
    fn default() -> Self {
       Self {
           surface_format_selector: default_format_score,
           present_mode: PresentMode::FifoKhr,
           present_operation: SurfaceTransform::IDENTITY,
           image_usage: ImageUsage::COLOR_ATTACHMENT,
           name: b"swapchain".into()
       }
    }
}
pub mod surface_format_selector {
    use crate::{Format, SurfaceFormatScore};

    pub extern "C" fn default_format_score(format: Format) -> SurfaceFormatScore{
        match format {
            Format::B8g8r8a8Unorm => 90,
            Format::R8g8b8a8Unorm => 80,
            Format::B8g8r8a8Srgb => 70,
            Format::R8g8b8a8Srgb => 60,
            _ => 0,
        }

    }
}

#[repr(u32)]
pub enum PresentMode {
    ImmediateKhr = 0,
    MailboxKhr = 1,
    FifoKhr = 2,
    FifoRelaxedKhr = 3,
    SharedDemandRefreshKhr = 1000111000,
    SharedContinuousRefreshKhr = 1000111001,
}
bitflags! {
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0x00000001;
        const TRANSFER_DST = 0x00000002;
        const SAMPLED = 0x00000004;
        const STORAGE = 0x00000008;
        const COLOR_ATTACHMENT = 0x00000010;
        const DEPTH_STENCIL_ATTACHMENT = 0x00000020;
        const TRANSIENT_ATTACHMENT = 0x00000040;
        const INPUT_ATTACHMENT = 0x00000080;
        const VIDEO_DECODE_DST_KHR = 0x00000400;
        const VIDEO_DECODE_SRC_KHR = 0x00000800;
        const VIDEO_DECODE_DPB_KHR = 0x00001000;
        const FRAGMENT_DENSITY_MAP_EXT = 0x00000200;
        const FRAGMENT_SHADING_RATE_ATTACHMENT_KHR = 0x00000100;
        const HOST_TRANSFER_EXT = 0x00400000;
        const VIDEO_ENCODE_DST_KHR = 0x00002000;
        const VIDEO_ENCODE_SRC_KHR = 0x00004000;
        const VIDEO_ENCODE_DPB_KHR = 0x00008000;
        const ATTACHMENT_FEEDBACK_LOOP_EXT = 0x00080000;
        const INVOCATION_MASK_HUAWEI = 0x00040000;
        const SAMPLE_WEIGHT_QCOM = 0x00100000;
        const SAMPLE_BLOCK_MATCH_QCOM = 0x00200000;
        const SHADING_RATE_IMAGE_NV = Self::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.bits();
    }
}

#[repr(u32)]
#[derive(Default)]
pub enum Format {
    #[default]
    Undefined = 0,
    R4g4UnormPack8 = 1,
    R4g4b4a4UnormPack16 = 2,
    B4g4r4a4UnormPack16 = 3,
    R5g6b5UnormPack16 = 4,
    B5g6r5UnormPack16 = 5,
    R5g5b5a1UnormPack16 = 6,
    B5g5r5a1UnormPack16 = 7,
    A1r5g5b5UnormPack16 = 8,
    R8Unorm = 9,
    R8Snorm = 10,
    R8Uscaled = 11,
    R8Sscaled = 12,
    R8Uint = 13,
    R8Sint = 14,
    R8Srgb = 15,
    R8g8Unorm = 16,
    R8g8Snorm = 17,
    R8g8Uscaled = 18,
    R8g8Sscaled = 19,
    R8g8Uint = 20,
    R8g8Sint = 21,
    R8g8Srgb = 22,
    R8g8b8Unorm = 23,
    R8g8b8Snorm = 24,
    R8g8b8Uscaled = 25,
    R8g8b8Sscaled = 26,
    R8g8b8Uint = 27,
    R8g8b8Sint = 28,
    R8g8b8Srgb = 29,
    B8g8r8Unorm = 30,
    B8g8r8Snorm = 31,
    B8g8r8Uscaled = 32,
    B8g8r8Sscaled = 33,
    B8g8r8Uint = 34,
    B8g8r8Sint = 35,
    B8g8r8Srgb = 36,
    R8g8b8a8Unorm = 37,
    R8g8b8a8Snorm = 38,
    R8g8b8a8Uscaled = 39,
    R8g8b8a8Sscaled = 40,
    R8g8b8a8Uint = 41,
    R8g8b8a8Sint = 42,
    R8g8b8a8Srgb = 43,
    B8g8r8a8Unorm = 44,
    B8g8r8a8Snorm = 45,
    B8g8r8a8Uscaled = 46,
    B8g8r8a8Sscaled = 47,
    B8g8r8a8Uint = 48,
    B8g8r8a8Sint = 49,
    B8g8r8a8Srgb = 50,
    A8b8g8r8UnormPack32 = 51,
    A8b8g8r8SnormPack32 = 52,
    A8b8g8r8UscaledPack32 = 53,
    A8b8g8r8SscaledPack32 = 54,
    A8b8g8r8UintPack32 = 55,
    A8b8g8r8SintPack32 = 56,
    A8b8g8r8SrgbPack32 = 57,
    A2r10g10b10UnormPack32 = 58,
    A2r10g10b10SnormPack32 = 59,
    A2r10g10b10UscaledPack32 = 60,
    A2r10g10b10SscaledPack32 = 61,
    A2r10g10b10UintPack32 = 62,
    A2r10g10b10SintPack32 = 63,
    A2b10g10r10UnormPack32 = 64,
    A2b10g10r10SnormPack32 = 65,
    A2b10g10r10UscaledPack32 = 66,
    A2b10g10r10SscaledPack32 = 67,
    A2b10g10r10UintPack32 = 68,
    A2b10g10r10SintPack32 = 69,
    R16Unorm = 70,
    R16Snorm = 71,
    R16Uscaled = 72,
    R16Sscaled = 73,
    R16Uint = 74,
    R16Sint = 75,
    R16Sfloat = 76,
    R16g16Unorm = 77,
    R16g16Snorm = 78,
    R16g16Uscaled = 79,
    R16g16Sscaled = 80,
    R16g16Uint = 81,
    R16g16Sint = 82,
    R16g16Sfloat = 83,
    R16g16b16Unorm = 84,
    R16g16b16Snorm = 85,
    R16g16b16Uscaled = 86,
    R16g16b16Sscaled = 87,
    R16g16b16Uint = 88,
    R16g16b16Sint = 89,
    R16g16b16Sfloat = 90,
    R16g16b16a16Unorm = 91,
    R16g16b16a16Snorm = 92,
    R16g16b16a16Uscaled = 93,
    R16g16b16a16Sscaled = 94,
    R16g16b16a16Uint = 95,
    R16g16b16a16Sint = 96,
    R16g16b16a16Sfloat = 97,
    R32Uint = 98,
    R32Sint = 99,
    R32Sfloat = 100,
    R32g32Uint = 101,
    R32g32Sint = 102,
    R32g32Sfloat = 103,
    R32g32b32Uint = 104,
    R32g32b32Sint = 105,
    R32g32b32Sfloat = 106,
    R32g32b32a32Uint = 107,
    R32g32b32a32Sint = 108,
    R32g32b32a32Sfloat = 109,
    R64Uint = 110,
    R64Sint = 111,
    R64Sfloat = 112,
    R64g64Uint = 113,
    R64g64Sint = 114,
    R64g64Sfloat = 115,
    R64g64b64Uint = 116,
    R64g64b64Sint = 117,
    R64g64b64Sfloat = 118,
    R64g64b64a64Uint = 119,
    R64g64b64a64Sint = 120,
    R64g64b64a64Sfloat = 121,
    B10g11r11UfloatPack32 = 122,
    E5b9g9r9UfloatPack32 = 123,
    D16Unorm = 124,
    X8D24UnormPack32 = 125,
    D32Sfloat = 126,
    S8Uint = 127,
    D16UnormS8Uint = 128,
    D24UnormS8Uint = 129,
    D32SfloatS8Uint = 130,
    Bc1RgbUnormBlock = 131,
    Bc1RgbSrgbBlock = 132,
    Bc1RgbaUnormBlock = 133,
    Bc1RgbaSrgbBlock = 134,
    Bc2UnormBlock = 135,
    Bc2SrgbBlock = 136,
    Bc3UnormBlock = 137,
    Bc3SrgbBlock = 138,
    Bc4UnormBlock = 139,
    Bc4SnormBlock = 140,
    Bc5UnormBlock = 141,
    Bc5SnormBlock = 142,
    Bc6hUfloatBlock = 143,
    Bc6hSfloatBlock = 144,
    Bc7UnormBlock = 145,
    Bc7SrgbBlock = 146,
    Etc2R8g8b8UnormBlock = 147,
    Etc2R8g8b8SrgbBlock = 148,
    Etc2R8g8b8a1UnormBlock = 149,
    Etc2R8g8b8a1SrgbBlock = 150,
    Etc2R8g8b8a8UnormBlock = 151,
    Etc2R8g8b8a8SrgbBlock = 152,
    EacR11UnormBlock = 153,
    EacR11SnormBlock = 154,
    EacR11g11UnormBlock = 155,
    EacR11g11SnormBlock = 156,
    Astc4x4UnormBlock = 157,
    Astc4x4SrgbBlock = 158,
    Astc5x4UnormBlock = 159,
    Astc5x4SrgbBlock = 160,
    Astc5x5UnormBlock = 161,
    Astc5x5SrgbBlock = 162,
    Astc6x5UnormBlock = 163,
    Astc6x5SrgbBlock = 164,
    Astc6x6UnormBlock = 165,
    Astc6x6SrgbBlock = 166,
    Astc8x5UnormBlock = 167,
    Astc8x5SrgbBlock = 168,
    Astc8x6UnormBlock = 169,
    Astc8x6SrgbBlock = 170,
    Astc8x8UnormBlock = 171,
    Astc8x8SrgbBlock = 172,
    Astc10x5UnormBlock = 173,
    Astc10x5SrgbBlock = 174,
    Astc10x6UnormBlock = 175,
    Astc10x6SrgbBlock = 176,
    Astc10x8UnormBlock = 177,
    Astc10x8SrgbBlock = 178,
    Astc10x10UnormBlock = 179,
    Astc10x10SrgbBlock = 180,
    Astc12x10UnormBlock = 181,
    Astc12x10SrgbBlock = 182,
    Astc12x12UnormBlock = 183,
    Astc12x12SrgbBlock = 184,
    G8b8g8r8422Unorm = 1000156000,
    B8g8r8g8422Unorm = 1000156001,
    G8B8R83plane420Unorm = 1000156002,
    G8B8r82plane420Unorm = 1000156003,
    G8B8R83plane422Unorm = 1000156004,
    G8B8r82plane422Unorm = 1000156005,
    G8B8R83plane444Unorm = 1000156006,
    R10x6UnormPack16 = 1000156007,
    R10x6g10x6Unorm2pack16 = 1000156008,
    R10x6g10x6b10x6a10x6Unorm4pack16 = 1000156009,
    G10x6b10x6g10x6r10x6422Unorm4pack16 = 1000156010,
    B10x6g10x6r10x6g10x6422Unorm4pack16 = 1000156011,
    G10x6B10x6R10x63plane420Unorm3pack16 = 1000156012,
    G10x6B10x6r10x62plane420Unorm3pack16 = 1000156013,
    G10x6B10x6R10x63plane422Unorm3pack16 = 1000156014,
    G10x6B10x6r10x62plane422Unorm3pack16 = 1000156015,
    G10x6B10x6R10x63plane444Unorm3pack16 = 1000156016,
    R12x4UnormPack16 = 1000156017,
    R12x4g12x4Unorm2pack16 = 1000156018,
    R12x4g12x4b12x4a12x4Unorm4pack16 = 1000156019,
    G12x4b12x4g12x4r12x4422Unorm4pack16 = 1000156020,
    B12x4g12x4r12x4g12x4422Unorm4pack16 = 1000156021,
    G12x4B12x4R12x43plane420Unorm3pack16 = 1000156022,
    G12x4B12x4r12x42plane420Unorm3pack16 = 1000156023,
    G12x4B12x4R12x43plane422Unorm3pack16 = 1000156024,
    G12x4B12x4r12x42plane422Unorm3pack16 = 1000156025,
    G12x4B12x4R12x43plane444Unorm3pack16 = 1000156026,
    G16b16g16r16422Unorm = 1000156027,
    B16g16r16g16422Unorm = 1000156028,
    G16B16R163plane420Unorm = 1000156029,
    G16B16r162plane420Unorm = 1000156030,
    G16B16R163plane422Unorm = 1000156031,
    G16B16r162plane422Unorm = 1000156032,
    G16B16R163plane444Unorm = 1000156033,
    G8B8r82plane444Unorm = 1000330000,
    G10x6B10x6r10x62plane444Unorm3pack16 = 1000330001,
    G12x4B12x4r12x42plane444Unorm3pack16 = 1000330002,
    G16B16r162plane444Unorm = 1000330003,
    A4r4g4b4UnormPack16 = 1000340000,
    A4b4g4r4UnormPack16 = 1000340001,
    Astc4x4SfloatBlock = 1000066000,
    Astc5x4SfloatBlock = 1000066001,
    Astc5x5SfloatBlock = 1000066002,
    Astc6x5SfloatBlock = 1000066003,
    Astc6x6SfloatBlock = 1000066004,
    Astc8x5SfloatBlock = 1000066005,
    Astc8x6SfloatBlock = 1000066006,
    Astc8x8SfloatBlock = 1000066007,
    Astc10x5SfloatBlock = 1000066008,
    Astc10x6SfloatBlock = 1000066009,
    Astc10x8SfloatBlock = 1000066010,
    Astc10x10SfloatBlock = 1000066011,
    Astc12x10SfloatBlock = 1000066012,
    Astc12x12SfloatBlock = 1000066013,
    Pvrtc12bppUnormBlockImg = 1000054000,
    Pvrtc14bppUnormBlockImg = 1000054001,
    Pvrtc22bppUnormBlockImg = 1000054002,
    Pvrtc24bppUnormBlockImg = 1000054003,
    Pvrtc12bppSrgbBlockImg = 1000054004,
    Pvrtc14bppSrgbBlockImg = 1000054005,
    Pvrtc22bppSrgbBlockImg = 1000054006,
    Pvrtc24bppSrgbBlockImg = 1000054007,
    R16g16Sfixed5Nv = 1000464000,
    A1b5g5r5UnormPack16Khr = 1000470000,
    A8UnormKhr = 1000470001,
}
#[derive(Clone, Copy)]
pub struct Swapchain(daxa_sys::daxa_Swapchain);

#[repr(C)]
pub struct ImageId(daxa_sys::daxa_ImageId);

impl Swapchain {
    pub fn acquire_next_image(&self) -> ImageId {
        unsafe {
            let mut image_id = MaybeUninit::uninit();
            daxa_sys::daxa_swp_acquire_next_image(self.0, image_id.as_mut_ptr());
            ImageId(image_id.assume_init())
        }
    }
    pub fn current_present_semaphore(&self) -> BinarySemaphore {
        unsafe {
            BinarySemaphore(daxa_sys::daxa_swp_current_present_semaphore(self.0).read())
        }
    }
    pub fn current_acquire_semaphore(&self) -> BinarySemaphore {
        unsafe {
            BinarySemaphore(daxa_sys::daxa_swp_current_acquire_semaphore(self.0).read())
        }
    }
    pub fn get_format(&self) -> Format {
        unsafe { mem::transmute(daxa_sys::daxa_swp_get_format(self.0))}
    }
}
// #[cfg(feature = "glsl")]
// pub struct GlslCompiler {
//     compiler: &'static glslang::Compiler,
//     root_paths: Vec<PathBuf>,
//     enable_debug_info: bool,
// }
//
// impl GlslCompiler {
//     pub fn new(root_paths: Vec<PathBuf>, enable_debug_info: bool) -> Self {
//         Self {
//             compiler: glslang::Compiler::acquire().expect("failed to acquire glslang"),
//             root_paths,
//             enable_debug_info
//         }
//     }
//     pub fn compile_raster_pipeline(&self, device: &Device, info: &RasterCompileInfo) -> daxa_sys::daxa_RasterPipeline {
//         let Device(device) = device;
//         todo!()
//     }
// }
//
// pub enum ShaderCompiler {
//     #[cfg(feature = "glsl")]
//     Glsl(GlslCompiler)
// }
//
// impl ShaderCompiler {
//     pub(crate) fn compile_raster_pipeline(&self, device: &Device, info: &RasterCompileInfo) -> daxa_sys::daxa_RasterPipeline {
//         match self {
//             ShaderCompiler::Glsl(compiler) => compiler.compile_raster_pipeline(device, info),
//         }
//     }
// }
//
// pub enum ShaderLanguage {
//     #[cfg(feature = "glsl")]
//     Glsl
// }
//
// impl ShaderLanguage {
//     pub(crate) fn new_compiler(&self, root_paths: Vec<PathBuf>, enable_debug_info: bool) -> ShaderCompiler {
//         match self {
//             Self::Glsl => ShaderCompiler::Glsl(GlslCompiler::new(root_paths, enable_debug_info)),
//         }
//     }
// }
//
// pub struct PipelineManagerInfo {
//     root_paths: Vec<PathBuf>,
//     language: ShaderLanguage,
//     enable_debug_info: bool,
// }
//
// pub struct PipelineManager {
//     device: daxa_sys::daxa_Device,
//     compiler: ShaderCompiler,
//     raster_pipelines: Vec<RasterPipeline>,
//     raster_pipeline_infos: Vec<RasterCompileInfo>,
//     raster_pipeline_handles: Vec<Handle<RasterPipeline>>,
// }
//
// impl PipelineManager {
//     fn new(device: &Device, info: PipelineManagerInfo) -> Self {
//         let Device(device) = device;
//         let PipelineManagerInfo { root_paths, language, enable_debug_info } = info;
//         let compiler = language.new_compiler(root_paths, enable_debug_info);
//         todo!()
//     }
//     fn add_raster_pipeline(&mut self, device: &Device,info: RasterCompileInfo) -> Handle<RasterPipeline> {
//         let pipeline = self.compiler.compile_raster_pipeline(&device, &info);
//         self.raster_pipelines.push(RasterPipeline(pipeline));
//         let handle = Handle(Arc::new(AtomicPtr::new(self.raster_pipelines.last_mut().unwrap())));
//         self.raster_pipeline_handles.push(handle.clone());
//         self.raster_pipeline_infos.push(info);
//         handle
//     }
// }

bitflags! {
    pub struct PipelineShaderStageCreateFlags: u32 {
        /// Provided by VK_VERSION_1_3
        const ALLOW_VARYING_SUBGROUP_SIZE = 0x00000001;
        /// Provided by VK_VERSION_1_3
        const REQUIRE_FULL_SUBGROUPS = 0x00000002;
        /// Provided by VK_EXT_subgroup_size_control
        const ALLOW_VARYING_SUBGROUP_SIZE_EXT = Self::ALLOW_VARYING_SUBGROUP_SIZE.bits();
        /// Provided by VK_EXT_subgroup_size_control
        const REQUIRE_FULL_SUBGROUPS_EXT = Self::REQUIRE_FULL_SUBGROUPS.bits();
    }
}

#[repr(C)]
pub struct ShaderInfo {
    pub byte_code: *const u32,
    pub byte_code_size: u32,
    pub create_flags: PipelineShaderStageCreateFlags,
    pub required_subgroup_size: Optional<u32>,
    pub entry_point: SmallString,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Never = 0,
    Less = 1,
    Equal = 2,
    LessOrEqual = 3,
    Greater = 4,
    NotEqual = 5,
    GreaterOrEqual = 6,
    Always = 7,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BlendFactor {
    Zero = 0,
    One = 1,
    SrcColor = 2,
    OneMinusSrcColor = 3,
    DstColor = 4,
    OneMinusDstColor = 5,
    SrcAlpha = 6,
    OneMinusSrcAlpha = 7,
    DstAlpha = 8,
    OneMinusDstAlpha = 9,
    ConstantColor = 10,
    OneMinusConstantColor = 11,
    ConstantAlpha = 12,
    OneMinusConstantAlpha = 13,
    SrcAlphaSaturate = 14,
    Src1Color = 15,
    OneMinusSrc1Color = 16,
    Src1Alpha = 17,
    OneMinusSrc1Alpha = 18,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum BlendOp {
    Add = 0,
    Subtract = 1,
    ReverseSubtract = 2,
    Min = 3,
    Max = 4,
}

bitflags! {
    pub struct ColorComponentFlags: u32 {
        const NONE = 0x00000000;
        const R = 0x00000001;
        const G = 0x00000002;
        const B = 0x00000004;
        const A = 0x00000008;
    }
}

#[repr(C)]
pub struct BlendInfo {
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOp,
    pub color_write_mask: ColorComponentFlags,
}

impl Default for BlendInfo {
    fn default() -> Self {
        Self {
            src_color_blend_factor: BlendFactor::One,
            dst_color_blend_factor: BlendFactor::Zero,
            color_blend_op: BlendOp::Add,
            src_alpha_blend_factor: BlendFactor::One,
            dst_alpha_blend_factor: BlendFactor::Zero,
            alpha_blend_op: BlendOp::Add,
            color_write_mask: ColorComponentFlags::R | ColorComponentFlags::G | ColorComponentFlags::B | ColorComponentFlags::A,
        }
    }
}


#[repr(C)]
#[derive(Default)]
pub struct RenderAttachment {
    pub format: Format,
    pub blend_info: Optional<BlendInfo>
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PrimitiveTopology {
    PointList = 0,
    LineList = 1,
    LineStrip = 2,
    TriangleList = 3,
    TriangleStrip = 4,
    TriangleFan = 5,
    LineListWithAdjacency = 6,
    LineStripWithAdjacency = 7,
    TriangleListWithAdjacency = 8,
    TriangleStripWithAdjacency = 9,
    PatchList = 10,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum PolygonMode {
    Fill = 0,
    Line = 1,
    Point = 2,
    FillRectangleNV = 1000153000,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum FrontFaceWinding {
    CounterClockwise = 0,
    Clockwise = 1,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ConservativeRasterizationMode {
    Disabled = 0,
    Overestimate = 1,
    Underestimate = 2
}

#[repr(C)]
struct ConservativeRasterInfo {
    mode: ConservativeRasterizationMode,
    size: f32,
}

#[repr(C)]
pub struct RasterizerInfo {
    primitive_topology: PrimitiveTopology,
    primitive_restart_enable: bool,
    polygon_mode: PolygonMode,
    face_culling: FaceCull,
    front_face_winding: FrontFaceWinding,
    depth_clamp_enable: bool,
    rasterizer_discard_enable: bool,
    depth_bias_enable: bool,
    depth_bias_constant_factor: f32,
    depth_bias_clamp: f32,
    depth_bias_slope_factor: f32,
    line_width: f32,
    conservative_raster_info: Optional<ConservativeRasterInfo>,
    static_state_sample_count: Optional<RasterizationSamples>,
}

#[repr(u32)]
pub enum RasterizationSamples {
E1 = 0x00000001,
E2 = 0x00000002,
E4 = 0x00000004,
E8 = 0x00000008,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum TessellationDomainOrigin {
    UpperLeft = 0,
    LowerLeft = 1,
}


bitflags! {
    #[repr(transparent)]
    pub struct FaceCull: u32 {
        const NONE = 0;
        const FRONT = 0x00000001;
        const BACK = 0x00000002;
        const FRONT_AND_BACK = Self::FRONT.bits() | Self::BACK.bits();
    }
}

#[repr(C)]
pub struct TessellationInfo {
    control_points: u32,
    origin: TessellationDomainOrigin
}

#[repr(C)]
pub struct FixedList<T, const C: usize> {
    data: [MaybeUninit<T>; C],
    len: u8
}

impl<T, const C: usize> Default for FixedList<T, C> {
    fn default() -> Self {
        Self {
            data: MaybeUninit::uninit_array(),
            len: 0,
        }
    }
}

impl<T, I: IntoIterator<Item = T>, const C: usize> From<I> for FixedList<T, C> {
    fn from(value: I) -> Self {
        let mut list = Self::default();
        for (i, t) in value.into_iter().enumerate().take(C) {
            list.data[i] = MaybeUninit::new(t);
            list.len = (i + 1) as u8;
        }
        list
    }
}

pub struct DepthTestInfo {
    pub depth_attachment_format: Format,
    pub enable_depth_write: bool,
    pub depth_test_compare_op: CompareOp,
    pub min_depth_bounds: f32,
    pub max_depth_bounds: f32,
}
#[repr(C)]
#[derive(Default)]
pub struct RasterPipelineInfo {
    pub mesh_shader_info: Optional<ShaderInfo>,
    pub vertex_shader_info: Optional<ShaderInfo>,
    pub tesselation_control_shader_info: Optional<ShaderInfo>,
    pub tesselation_evaluation_shader_info: Optional<ShaderInfo>,
    pub fragment_shader_info: Optional<ShaderInfo>,
    pub task_shader_info: Optional<ShaderInfo>,
    pub color_attachments: FixedList<RenderAttachment, 8>,
    pub depth_test: Optional<DepthTestInfo>,
    pub tessellation: Optional<TessellationInfo>,
    pub raster: RasterizerInfo,
    pub push_constant_size: u32,
    pub name: SmallString
}

impl Default for RasterizerInfo {
    fn default() -> Self {
        Self {
            primitive_topology: PrimitiveTopology::TriangleList,
            primitive_restart_enable: false,
            polygon_mode: PolygonMode::Fill,
            face_culling: FaceCull::BACK,
            front_face_winding: FrontFaceWinding::Clockwise,
            depth_clamp_enable: false,
            rasterizer_discard_enable: false,
            depth_bias_enable: false,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            conservative_raster_info: Optional::default(),
            static_state_sample_count: Optional::default(),
        }
    }
}

impl Default for TessellationInfo {
    fn default() -> Self {
        Self {
            control_points: 0,
            origin: TessellationDomainOrigin::UpperLeft,
        }
    }
}

impl Default for SmallString {
    fn default() -> Self {
        Self(daxa_sys::daxa_SmallString {
            data: [0; SMALL_STRING_CAPACITY],
            size: 0,
        })
    }
}

pub struct RasterPipeline(daxa_sys::daxa_RasterPipeline);

unsafe impl Send for RasterPipeline {}
unsafe impl Sync for RasterPipeline {}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct ImageViewId(daxa_sys::daxa_ImageViewId);

#[repr(C)]
#[derive(Copy, Clone)]
pub struct AttachmentResolveInfo {
    pub mode: ResolveModeFlag,
    pub image: ImageViewId,
    pub layout: ImageLayout,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ImageLayout {
    Undefined = 0,
    General = 1,
    ColorAttachmentOptimal = 2,
    DepthStencilAttachmentOptimal = 3,
    DepthStencilReadOnlyOptimal = 4,
    ShaderReadOnlyOptimal = 5,
    TransferSrcOptimal = 6,
    TransferDstOptimal = 7,
    Preinitialized = 8,
    DepthReadOnlyStencilAttachmentOptimal = 1000117000,
    DepthAttachmentStencilReadOnlyOptimal = 1000117001,
    DepthAttachmentOptimal = 1000241000,
    DepthReadOnlyOptimal = 1000241001,
    StencilAttachmentOptimal = 1000241002,
    StencilReadOnlyOptimal = 1000241003,
    ReadOnlyOptimal = 1000314000,
    AttachmentOptimal = 1000314001,
    PresentSrcKhr = 1000001002,
    VideoDecodeDstKhr = 1000024000,
    VideoDecodeSrcKhr = 1000024001,
    VideoDecodeDpbKhr = 1000024002,
    SharedPresentKhr = 1000111000,
    FragmentDensityMapOptimalExt = 1000218000,
    FragmentShadingRateAttachmentOptimalKhr = 1000164003,
    RenderingLocalReadKhr = 1000232000,
    VideoEncodeDstKhr = 1000299000,
    VideoEncodeSrcKhr = 1000299001,
    VideoEncodeDpbKhr = 1000299002,
    AttachmentFeedbackLoopOptimalExt = 1000339000
}

#[repr(C)]
#[derive(Clone)]
pub struct RenderAttachmentInfo {
    pub image_view: ImageViewId,
    pub layout: ImageLayout,
    pub load_op: AttachmentLoadOp,
    pub store_op: AttachmentStoreOp,
    pub clear_value: Optional<ClearValue>,
    pub resolve: Optional<AttachmentResolveInfo>,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AttachmentStoreOp {
    Store = 0,
    DontCare = 1,
    None = 1000301000,
}

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum AttachmentLoadOp {
    Load = 0,
    Clear = 1,
    DontCare = 2,
    NoneKhr = 1000400000,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub union ClearValue {
    pub color: ClearColorValue,
    depth_stencil: ClearDepthStencilValue
}

#[repr(C)]
#[derive( Copy, Clone)]
pub union ClearColorValue {
    pub float32: [f32; 4],
    pub int32: [i32; 4],
    uint32: [u32; 4],
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ClearDepthStencilValue {
    depth: f32,
    stencil: u32,
}