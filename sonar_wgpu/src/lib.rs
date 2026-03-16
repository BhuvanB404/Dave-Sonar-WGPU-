//! sonar_wgpu — vendor-agnostic GPU sonar compute backend
//! Optimized version: single command encoder, fused scatter+sum shader,
//! precomputed twiddle table FFT, eliminated per-frame heap allocation.

use std::mem::size_of;
use std::str::FromStr;
use std::sync::mpsc;
use std::time::{Duration, Instant};
use std::cell::UnsafeCell;

// Shader sources embedded at compile time.
const SCATTER_SUM_FUSED_WGSL: &str = include_str!("shaders/scatter_sum_fused.wgsl");
const MATMUL_DUAL_WGSL: &str      = include_str!("shaders/matmul_dual.wgsl");
const FFT_TWIDDLE_WGSL: &str      = include_str!("shaders/fft_twiddle.wgsl");
const BLUESTEIN_PREPARE_WGSL: &str = include_str!("shaders/bluestein_prepare.wgsl");
const BLUESTEIN_FINALIZE_WGSL: &str = include_str!("shaders/bluestein_finalize.wgsl");
const POINTWISE_MUL_WGSL: &str    = include_str!("shaders/pointwise_mul.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct ScatterConfig {
    nBeams:       u32,
    nRays:        u32,
    nFreq:        u32,
    raySkips:     u32,
    soundSpeed:   f32,
    delta_f:      f32,
    sourceTerm:   f32,
    attenuation:  f32,
    areaScaler:   f32,
    maxDistance:  f32,
    frameSeed:    u32,
    _pad:         u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MatConfig { m: u32, k: u32, n: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct FftTwConfig { n: u32, nBeams: u32, inverse: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct PrepareConfig { nBeams: u32, nFreq: u32, m: u32, _pad: u32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(non_snake_case)]
struct FinalizeConfig { nBeams: u32, nFreq: u32, m: u32, delta_f: f32 }

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MulConfig { n: u32, m: u32, _pad: [u32; 2] }

#[repr(C)]
#[derive(Copy, Clone, Default)]
pub struct SonarWgpuRunTimings {
    pub upload_us:                u64,
    pub scatter_us:               u64,
    pub beam_sum_us:              u64,
    pub readback_us:              u64,
    pub fallback_scatter_sum_us:  u64,
    pub correction_us:            u64,
    pub dft_us:                   u64,
    pub total_us:                 u64,
}

struct PipelineObj {
    pipeline: wgpu::ComputePipeline,
}

impl PipelineObj {
    fn new(device: &wgpu::Device, src: &str, label: &str) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(label),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(label),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline }
    }

    fn bind_group(&self, device: &wgpu::Device, entries: &[wgpu::BindGroupEntry]) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries,
        })
    }
}

pub struct SonarWgpuContext {
    device: wgpu::Device,
    queue:  wgpu::Queue,

    // Pipelines
    scatter_sum_fused_pl:  PipelineObj,
    matmul_dual_pl:        PipelineObj,
    fft_twiddle_pl:        PipelineObj,
    bluestein_prepare_pl:  PipelineObj,
    bluestein_finalize_pl: PipelineObj,
    pointwise_mul_pl:      PipelineObj,

    // Input buffers
    depth_buf:         wgpu::Buffer,
    normals_buf:       wgpu::Buffer,
    reflect_buf:       wgpu::Buffer,
    beam_corrector_buf: wgpu::Buffer,

    // Intermediate beam buffers
    beam_re_buf: wgpu::Buffer,
    beam_im_buf: wgpu::Buffer,
    cor_re_buf:  wgpu::Buffer,
    cor_im_buf:  wgpu::Buffer,

    // Bluestein FFT buffers
    chirp_conj_re_buf: wgpu::Buffer,
    chirp_conj_im_buf: wgpu::Buffer,
    chirp_fft_re_buf:  wgpu::Buffer,
    chirp_fft_im_buf:  wgpu::Buffer,
    twiddle_re_buf:    wgpu::Buffer,
    twiddle_im_buf:    wgpu::Buffer,
    a_re_buf:          wgpu::Buffer,
    a_im_buf:          wgpu::Buffer,
    c_re_buf:          wgpu::Buffer,
    c_im_buf:          wgpu::Buffer,

    staging_buf: wgpu::Buffer,

    // Uniform config buffers
    scatter_cfg_buf:  wgpu::Buffer,
    mat_cfg_buf:      wgpu::Buffer,
    fft_fwd_cfg_buf:  wgpu::Buffer,
    fft_inv_cfg_buf:  wgpu::Buffer,
    prepare_cfg_buf:  wgpu::Buffer,
    finalize_cfg_buf: wgpu::Buffer,
    mul_cfg_buf:      wgpu::Buffer,

    // Pre-created bind groups
    scatter_sum_bg: wgpu::BindGroup,
    matmul_dual_bg: wgpu::BindGroup,
    prepare_bg:     wgpu::BindGroup,
    fft_fwd_a_bg:   wgpu::BindGroup,
    pointwise_mul_bg: wgpu::BindGroup,
    fft_inv_c_bg:   wgpu::BindGroup,
    finalize_bg:    wgpu::BindGroup,

    n_beams:    usize,
    n_rays:     usize,
    n_rays_eff: usize,
    n_freq:     usize,
    ray_skips:  usize,
    fft_m:      usize,

    // Reusable scratch buffer — avoids per-frame heap allocation (UnsafeCell for interior mutability)
    bc_scratch: UnsafeCell<Vec<f32>>,

    // Timestamp queries for proper GPU profiling
    query_set: Option<wgpu::QuerySet>,
    query_resolve_buf: Option<wgpu::Buffer>,
    query_staging_buf: Option<wgpu::Buffer>,
    timestamp_period: f32,
}

pub type SonarWgpu = SonarWgpuContext;

fn parse_backends_from_env() -> wgpu::Backends {
    let Some(raw) = std::env::var("SONAR_WGPU_BACKENDS").ok() else {
        return wgpu::Backends::VULKAN;
    };
    let mut out = wgpu::Backends::empty();
    for token in raw.split(',').map(|s| s.trim().to_ascii_lowercase()) {
        match token.as_str() {
            "vulkan"              => out |= wgpu::Backends::VULKAN,
            "metal"               => out |= wgpu::Backends::METAL,
            "dx12"                => out |= wgpu::Backends::DX12,
            "gl"|"gles"|"opengl"  => out |= wgpu::Backends::GL,
            "browser"             => out |= wgpu::Backends::BROWSER_WEBGPU,
            "primary"             => out |= wgpu::Backends::PRIMARY,
            "secondary"           => out |= wgpu::Backends::SECONDARY,
            _ => {}
        }
    }
    if out.is_empty() { wgpu::Backends::VULKAN } else { out }
}

fn env_truthy(var: &str, default: bool) -> bool {
    match std::env::var(var) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no"),
        Err(_) => default,
    }
}

fn select_adapter(instance: &wgpu::Instance, backends: wgpu::Backends) -> Option<wgpu::Adapter> {
    let mut adapters: Vec<wgpu::Adapter> = instance.enumerate_adapters(backends).into_iter().collect();
    if adapters.is_empty() { return None; }

    if let Ok(name_filter) = std::env::var("SONAR_WGPU_ADAPTER_NAME") {
        let needle = name_filter.to_ascii_lowercase();
        if let Some(idx) = adapters.iter().position(|a| a.get_info().name.to_ascii_lowercase().contains(&needle)) {
            return Some(adapters.remove(idx));
        }
    }
    if env_truthy("SONAR_WGPU_PREFER_NVIDIA", true) {
        if let Some(idx) = adapters.iter().position(|a| {
            let info = a.get_info();
            info.vendor == 0x10de || info.name.to_ascii_lowercase().contains("nvidia")
        }) {
            return Some(adapters.remove(idx));
        }
    }
    adapters.into_iter().next()
}

fn parse_u64_env(var: &str, default: u64) -> u64 {
    std::env::var(var).ok().and_then(|v| u64::from_str(v.trim()).ok()).unwrap_or(default)
}

fn duration_us(d: Duration) -> u64 { d.as_micros() as u64 }

fn next_power_of_2(n: usize) -> usize {
    let mut p = 1;
    while p < n { p *= 2; }
    p
}

fn make_twiddle(n: usize) -> (Vec<f32>, Vec<f32>) {
    use std::f32::consts::PI;
    let half = n / 2;
    let mut re = vec![0.0f32; half];
    let mut im = vec![0.0f32; half];
    for k in 0..half {
        let angle = -2.0 * PI * k as f32 / n as f32;
        re[k] = angle.cos();
        im[k] = angle.sin();
    }
    (re, im)
}

impl SonarWgpuContext {
    pub fn new(
        n_beams: usize, n_rays: usize, n_freq: usize, ray_skips: usize,
        _sound_speed: f32, _delta_f: f32, _source_term: f32,
        _attenuation: f32, _area_scaler: f32, _max_distance: f32,
    ) -> Result<Self, String> {
        pollster::block_on(Self::new_async(n_beams, n_rays, n_freq, ray_skips))
    }

    async fn new_async(
        n_beams: usize, n_rays: usize, n_freq: usize, ray_skips: usize,
    ) -> Result<Self, String> {
        if n_beams == 0 || n_rays == 0 || n_freq == 0 { return Err("Dimensions must be > 0".to_string()); }
        if ray_skips == 0 { return Err("ray_skips must be > 0".to_string()); }
        if n_rays < ray_skips { return Err("n_rays must be >= ray_skips".to_string()); }

        let backends = parse_backends_from_env();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor { backends, ..Default::default() });

        let adapter = if let Some(a) = select_adapter(&instance, backends) {
            a
        } else {
            instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }).await.ok_or("No GPU adapter found")?
        };

        let info = adapter.get_info();
        eprintln!("[sonar_wgpu] Adapter: {} ({:?}, vendor=0x{:04x})", info.name, info.backend, info.vendor);

        let adapter_features = adapter.features();
        let supports_timestamps =
            adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY)
            && env_truthy("SONAR_WGPU_ENABLE_TIMESTAMPS", false);
        let mut required_features = wgpu::Features::empty();
        if supports_timestamps { required_features |= wgpu::Features::TIMESTAMP_QUERY; }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            }, None)
            .await.map_err(|e| format!("request_device failed: {e}"))?;

        let n_rays_eff = n_rays / ray_skips;
        let fft_m = next_power_of_2(2 * n_freq - 1);

        macro_rules! sbuf {
            ($n:expr, $name:expr) => {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some($name),
                    size: ($n * size_of::<f32>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            };
        }
        macro_rules! ubuf {
            ($t:ty, $name:expr) => {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some($name),
                    size: size_of::<$t>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            };
        }

        let total_rays  = n_rays * n_beams;
        let beam_total  = n_beams * n_freq;

        let depth_buf         = sbuf!(total_rays, "depth");
        let normals_buf       = sbuf!(total_rays * 3, "normals");
        let reflect_buf       = sbuf!(total_rays, "reflectivity");
        let beam_corrector_buf = sbuf!(n_beams * n_beams, "beam_corrector");
        let beam_re_buf       = sbuf!(beam_total, "beam_re");
        let beam_im_buf       = sbuf!(beam_total, "beam_im");
        let cor_re_buf        = sbuf!(beam_total, "cor_re");
        let cor_im_buf        = sbuf!(beam_total, "cor_im");

        let chirp_conj_re_buf  = sbuf!(n_freq, "chirp_conj_re");
        let chirp_conj_im_buf  = sbuf!(n_freq, "chirp_conj_im");
        let chirp_fft_re_buf   = sbuf!(fft_m, "chirp_fft_re");
        let chirp_fft_im_buf   = sbuf!(fft_m, "chirp_fft_im");
        let twiddle_re_buf     = sbuf!(fft_m / 2, "twiddle_re");
        let twiddle_im_buf     = sbuf!(fft_m / 2, "twiddle_im");
        let a_re_buf           = sbuf!(n_beams * fft_m, "a_re");
        let a_im_buf           = sbuf!(n_beams * fft_m, "a_im");
        let c_re_buf           = sbuf!(n_beams * fft_m, "c_re");
        let c_im_buf           = sbuf!(n_beams * fft_m, "c_im");

        let staging_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: (beam_total * 2 * size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let scatter_cfg_buf  = ubuf!(ScatterConfig, "scatter_cfg");
        let mat_cfg_buf      = ubuf!(MatConfig,     "mat_cfg");
        let fft_fwd_cfg_buf  = ubuf!(FftTwConfig,   "fft_fwd_cfg");
        let fft_inv_cfg_buf  = ubuf!(FftTwConfig,   "fft_inv_cfg");
        let prepare_cfg_buf  = ubuf!(PrepareConfig,  "prepare_cfg");
        let finalize_cfg_buf = ubuf!(FinalizeConfig, "finalize_cfg");
        let mul_cfg_buf      = ubuf!(MulConfig,      "mul_cfg");

        // Pipelines
        let scatter_sum_fused_pl  = PipelineObj::new(&device, SCATTER_SUM_FUSED_WGSL, "scatter_sum_fused");
        let matmul_dual_pl        = PipelineObj::new(&device, MATMUL_DUAL_WGSL,       "matmul_dual");
        let fft_twiddle_pl        = PipelineObj::new(&device, FFT_TWIDDLE_WGSL,       "fft_twiddle");
        let bluestein_prepare_pl  = PipelineObj::new(&device, BLUESTEIN_PREPARE_WGSL, "bluestein_prepare");
        let bluestein_finalize_pl = PipelineObj::new(&device, BLUESTEIN_FINALIZE_WGSL,"bluestein_finalize");
        let pointwise_mul_pl      = PipelineObj::new(&device, POINTWISE_MUL_WGSL,     "pointwise_mul");

        // Bind groups
        let scatter_sum_bg = scatter_sum_fused_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: normals_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: reflect_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: beam_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: beam_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: scatter_cfg_buf.as_entire_binding() },
        ]);

        let matmul_dual_bg = matmul_dual_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: beam_corrector_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: beam_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: beam_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: cor_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: cor_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: mat_cfg_buf.as_entire_binding() },
        ]);

        let prepare_bg = bluestein_prepare_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: cor_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: cor_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: chirp_conj_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: chirp_conj_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: a_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: a_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: prepare_cfg_buf.as_entire_binding() },
        ]);

        let fft_fwd_a_bg = fft_twiddle_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: a_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: twiddle_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: twiddle_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: fft_fwd_cfg_buf.as_entire_binding() },
        ]);

        let pointwise_mul_bg = pointwise_mul_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: a_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: a_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: chirp_fft_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: chirp_fft_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: c_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: c_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: mul_cfg_buf.as_entire_binding() },
        ]);

        let fft_inv_c_bg = fft_twiddle_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: c_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: c_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: twiddle_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: twiddle_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: fft_inv_cfg_buf.as_entire_binding() },
        ]);

        let finalize_bg = bluestein_finalize_pl.bind_group(&device, &[
            wgpu::BindGroupEntry { binding: 0, resource: c_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: c_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: chirp_conj_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: chirp_conj_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: beam_re_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 5, resource: beam_im_buf.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 6, resource: finalize_cfg_buf.as_entire_binding() },
        ]);

        // ---- Pre-write static uniform configs ----
        let mat_cfg = MatConfig { m: n_beams as u32, k: n_beams as u32, n: n_freq as u32, _pad: 0 };
        queue.write_buffer(&mat_cfg_buf, 0, bytemuck::bytes_of(&mat_cfg));

        let fft_fwd_cfg = FftTwConfig { n: fft_m as u32, nBeams: n_beams as u32, inverse: 0, _pad: 0 };
        queue.write_buffer(&fft_fwd_cfg_buf, 0, bytemuck::bytes_of(&fft_fwd_cfg));

        let fft_inv_cfg = FftTwConfig { n: fft_m as u32, nBeams: n_beams as u32, inverse: 1, _pad: 0 };
        queue.write_buffer(&fft_inv_cfg_buf, 0, bytemuck::bytes_of(&fft_inv_cfg));

        let prepare_cfg = PrepareConfig { nBeams: n_beams as u32, nFreq: n_freq as u32, m: fft_m as u32, _pad: 0 };
        queue.write_buffer(&prepare_cfg_buf, 0, bytemuck::bytes_of(&prepare_cfg));

        let mul_cfg = MulConfig { n: (n_beams * fft_m) as u32, m: fft_m as u32, _pad: [0; 2] };
        queue.write_buffer(&mul_cfg_buf, 0, bytemuck::bytes_of(&mul_cfg));

        // ---- Precompute chirp + twiddle on CPU ----
        Self::precompute_static_data(
            &device, &queue,
            &chirp_conj_re_buf, &chirp_conj_im_buf,
            &chirp_fft_re_buf, &chirp_fft_im_buf,
            &twiddle_re_buf, &twiddle_im_buf,
            n_freq, fft_m,
        );

        let timestamp_period = queue.get_timestamp_period();
        let (query_set, query_resolve_buf, query_staging_buf) = if supports_timestamps {
            let qs = device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("timestamps"),
                count: 14,
                ty: wgpu::QueryType::Timestamp,
            });
            let qr = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query_resolve"),
                size: 14 * 8, // 14 u64 timestamps
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::QUERY_RESOLVE,
                mapped_at_creation: false,
            });
            let qst = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("query_staging"),
                size: 14 * 8,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            (Some(qs), Some(qr), Some(qst))
        } else {
            (None, None, None)
        };

        let bc_scratch = UnsafeCell::new(vec![0.0f32; n_beams * n_beams]);

        Ok(Self {
            device, queue,
            scatter_sum_fused_pl, matmul_dual_pl, fft_twiddle_pl,
            bluestein_prepare_pl, bluestein_finalize_pl, pointwise_mul_pl,
            depth_buf, normals_buf, reflect_buf, beam_corrector_buf,
            beam_re_buf, beam_im_buf, cor_re_buf, cor_im_buf,
            chirp_conj_re_buf, chirp_conj_im_buf,
            chirp_fft_re_buf, chirp_fft_im_buf,
            twiddle_re_buf, twiddle_im_buf,
            a_re_buf, a_im_buf, c_re_buf, c_im_buf,
            staging_buf,
            scatter_cfg_buf, mat_cfg_buf,
            fft_fwd_cfg_buf, fft_inv_cfg_buf,
            prepare_cfg_buf, finalize_cfg_buf, mul_cfg_buf,
            scatter_sum_bg, matmul_dual_bg, prepare_bg,
            fft_fwd_a_bg, pointwise_mul_bg, fft_inv_c_bg, finalize_bg,
            n_beams, n_rays, n_rays_eff, n_freq, ray_skips, fft_m,
            bc_scratch,
            query_set, query_resolve_buf, query_staging_buf, timestamp_period,
        })
    }

    fn precompute_static_data(
        device: &wgpu::Device, queue: &wgpu::Queue,
        chirp_conj_re: &wgpu::Buffer, chirp_conj_im: &wgpu::Buffer,
        chirp_fft_re: &wgpu::Buffer, chirp_fft_im: &wgpu::Buffer,
        tw_re_buf: &wgpu::Buffer, tw_im_buf: &wgpu::Buffer,
        n_freq: usize, fft_m: usize,
    ) {
        use std::f32::consts::PI;

        // Chirp sequence 
        let mut c_re = vec![0.0f32; n_freq];
        let mut c_im = vec![0.0f32; n_freq];
        let mut cc_re = vec![0.0f32; n_freq];
        let mut cc_im = vec![0.0f32; n_freq];
        for n in 0..n_freq {
            let angle = PI * (n as f32) * (n as f32) / n_freq as f32;
            c_re[n]  =  angle.cos();
            c_im[n]  =  angle.sin();
            cc_re[n] =  angle.cos();
            cc_im[n] = -angle.sin();
        }
        queue.write_buffer(chirp_conj_re, 0, bytemuck::cast_slice(&cc_re));
        queue.write_buffer(chirp_conj_im, 0, bytemuck::cast_slice(&cc_im));

        // Zero-padded chirp for FFT
        let mut p_re = vec![0.0f32; fft_m];
        let mut p_im = vec![0.0f32; fft_m];
        for i in 0..n_freq { p_re[i] = c_re[i]; p_im[i] = c_im[i]; }
        for i in (fft_m - n_freq + 1)..fft_m {
            let idx = fft_m - i;
            p_re[i] = c_rew[n] = e^{j π n² / N}[idx];
            p_im[i] = c_im[idx];
        }
        Self::cpu_radix2_fft(&mut p_re, &mut p_im, false);
        queue.write_buffer(chirp_fft_re, 0, bytemuck::cast_slice(&p_re));
        queue.write_buffer(chirp_fft_im, 0, bytemuck::cast_slice(&p_im));

        // Precomputed twiddle factors for fft_m-point FFT
        let (tw_re, tw_im) = make_twiddle(fft_m);
        queue.write_buffer(tw_re_buf, 0, bytemuck::cast_slice(&tw_re));
        queue.write_buffer(tw_im_buf, 0, bytemuck::cast_slice(&tw_im));

        queue.submit(std::iter::empty());
        device.poll(wgpu::Maintain::Wait);
    }

    fn cpu_radix2_fft(re: &mut [f32], im: &mut [f32], inverse: bool) {
        let n = re.len();
        assert!(n.is_power_of_two());
        let num_stages = n.trailing_zeros() as usize;

        // Bit-reversal permutation
        for i in 0..n {
            let mut j = 0;
            for k in 0..num_stages {
                if (i >> k) & 1 != 0 {
                    j |= 1 << (num_stages - 1 - k);
                }
            }
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        let sign: f32 = if inverse { 1.0 } else { -1.0 };
        for stage in 0..num_stages {
            let half_size = 1 << stage;
            let full_size = half_size << 1;
            for k in 0..(n / 2) {
                let group = k / half_size;
                let pair  = k % half_size;
                let i = group * full_size + pair;
                let j = i + half_size;
                let angle  = sign * 2.0 * std::f32::consts::PI * (pair as f32) / (full_size as f32);
                let tw_re = angle.cos();
                let tw_im = angle.sin();
                let a_re = re[i]; let a_im = im[i];
                let b_re = re[j]; let b_im = im[j];
                let tb_re = b_re * tw_re - b_im * tw_im;
                let tb_im = b_re * tw_im + b_im * tw_re;
                re[i] = a_re + tb_re; im[i] = a_im + tb_im;
                re[j] = a_re - tb_re; im[j] = a_im - tb_im;
            }
        }
        if inverse {
            let scale = 1.0 / n as f32;
            for x in re.iter_mut() { *x *= scale; }
            for x in im.iter_mut() { *x *= scale; }
        }
    }

    /// Simplified run method for benchmarking with default parameters
    pub fn run_simple(
        &self,
        depth: &[f32], normals: &[f32], reflectivity: &[f32],
        out_real: &mut [f32], out_imag: &mut [f32],
    ) -> Result<SonarWgpuRunTimings, String> {
        let beam_corrector = vec![1.0f32; self.n_beams * self.n_beams];
        self.run(depth, normals, reflectivity, &beam_corrector,
            1500.0, 16000.0, 50000.0, 1.0, 0.1, 1.0, 200.0,
            self.n_beams as f32, 0, out_real, out_imag)
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(non_snake_case)]
    pub fn run(
        &self,
        depth: &[f32], normals: &[f32], reflectivity: &[f32], beamCorrector: &[f32],
        soundSpeed: f32, bandwidth: f32, _sonarFreq: f32,
        sourceLevel: f32, attenuation: f32, areaScaler: f32, maxDistance: f32,
        beamCorrectorSum: f32, frameSeed: u32,
        out_real: &mut [f32], out_imag: &mut [f32],
    ) -> Result<SonarWgpuRunTimings, String> {
        // SAFETY: bc_scratch is only mutated here, never aliased while run() is active.
        let bc_scratch = unsafe { &mut *self.bc_scratch.get() };
        self.run_inner(
            depth, normals, reflectivity, beamCorrector,
            soundSpeed, bandwidth, sourceLevel, attenuation, areaScaler, maxDistance,
            beamCorrectorSum, frameSeed, out_real, out_imag, bc_scratch,
        )
    }

    #[allow(non_snake_case)]
    fn run_inner(
        &self,
        depth: &[f32], normals: &[f32], reflectivity: &[f32], beamCorrector: &[f32],
        soundSpeed: f32, bandwidth: f32,
        sourceLevel: f32, attenuation: f32, areaScaler: f32, maxDistance: f32,
        beamCorrectorSum: f32, frameSeed: u32,
        out_real: &mut [f32], out_imag: &mut [f32],
        bc_scratch: &mut Vec<f32>,
    ) -> Result<SonarWgpuRunTimings, String> {
        let total_start = Instant::now();

        // Input validation
        let exp = self.n_rays * self.n_beams;
        if depth.len() != exp { return Err(format!("depth len mismatch: {} vs {}", depth.len(), exp)); }
        if reflectivity.len() != exp { return Err(format!("reflectivity len mismatch")); }
        if normals.len() != exp * 3 { return Err(format!("normals len mismatch")); }
        if beamCorrector.len() != self.n_beams * self.n_beams { return Err("beam_corrector len mismatch".to_string()); }
        let out_n = self.n_beams * self.n_freq;
        if out_real.len() != out_n || out_imag.len() != out_n { return Err("output len mismatch".to_string()); }
        if !soundSpeed.is_finite() || soundSpeed <= 0.0 { return Err("sound_speed invalid".to_string()); }
        if !bandwidth.is_finite() || bandwidth <= 0.0 { return Err("bandwidth invalid".to_string()); }

        let q = &self.queue;
        let d = &self.device;
        let delta_f     = bandwidth / self.n_freq as f32;
        let pref        = 1e-6_f32;
        let sourceTerm = (10.0_f32.powf(sourceLevel / 10.0)).sqrt() * pref;
        let mut timings = SonarWgpuRunTimings::default();

        // ============ UPLOAD ============
        let t_upload = Instant::now();
        q.write_buffer(&self.depth_buf,   0, bytemuck::cast_slice(depth));
        q.write_buffer(&self.normals_buf, 0, bytemuck::cast_slice(normals));
        q.write_buffer(&self.reflect_buf, 0, bytemuck::cast_slice(reflectivity));

        // Normalize beam corrector in-place into pre-allocated scratch — no heap alloc
        let norm = if beamCorrectorSum.abs() > 1e-12 { 1.0 / beamCorrectorSum } else { 1.0 };
        bc_scratch.iter_mut().zip(beamCorrector.iter()).for_each(|(dst, &src)| *dst = src * norm);
        q.write_buffer(&self.beam_corrector_buf, 0, bytemuck::cast_slice(bc_scratch));

        // Dynamic configs
        let scatter_cfg = ScatterConfig {
            nBeams: self.n_beams as u32, nRays: self.n_rays as u32,
            nFreq: self.n_freq as u32, raySkips: self.ray_skips as u32,
            soundSpeed, delta_f, sourceTerm, attenuation, areaScaler, maxDistance,
            frameSeed, _pad: 0,
        };
        q.write_buffer(&self.scatter_cfg_buf,  0, bytemuck::bytes_of(&scatter_cfg));
        let finalize_cfg = FinalizeConfig {
            nBeams: self.n_beams as u32, nFreq: self.n_freq as u32,
            m: self.fft_m as u32, delta_f,
        };
        q.write_buffer(&self.finalize_cfg_buf, 0, bytemuck::bytes_of(&finalize_cfg));

        timings.upload_us = duration_us(t_upload.elapsed()).max(1);

        // ============ SINGLE FUSED GPU COMMAND ENCODER (stages 1-4) ============
        let t_gpu = Instant::now();
        let mut enc = d.create_command_encoder(&Default::default());

        macro_rules! pass {
            ($enc:expr, $label:expr, $idx:expr) => {{
                let timestamp_writes = self.query_set.as_ref().map(|qs| wgpu::ComputePassTimestampWrites {
                    query_set: qs,
                    beginning_of_pass_write_index: Some($idx * 2),
                    end_of_pass_write_index: Some($idx * 2 + 1),
                });
                $enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some($label),
                    timestamp_writes,
                })
            }}
        }

        // Stage 1+2: Fused scatter + beam sum
        {
            let t_scatter = Instant::now();
            let mut pass = pass!(enc, "scatter_sum", 0);
            pass.set_pipeline(&self.scatter_sum_fused_pl.pipeline);
            pass.set_bind_group(0, &self.scatter_sum_bg, &[]);
            // dispatch: (n_freq, n_beams, 1) — one workgroup per (beam, freq) pair
            pass.dispatch_workgroups(self.n_freq as u32, self.n_beams as u32, 1);
            drop(pass);
            timings.scatter_us  = duration_us(t_scatter.elapsed()).max(1);
            timings.beam_sum_us = 1; // fused — attributed to scatter
        }

        // Stage 3: Correction (dual matmul)
        {
            let t_cor = Instant::now();
            let mut pass = pass!(enc, "matmul_dual", 1);
            pass.set_pipeline(&self.matmul_dual_pl.pipeline);
            pass.set_bind_group(0, &self.matmul_dual_bg, &[]);
            let wx = (self.n_freq as u32 + 15) / 16;
            let wy = (self.n_beams as u32 + 15) / 16;
            pass.dispatch_workgroups(wx, wy, 1);
            drop(pass);
            timings.correction_us = duration_us(t_cor.elapsed()).max(1);
        }

        // Stage 4: Bluestein FFT pipeline 
        {
            let t_dft = Instant::now();
            // 4a: Prepare 
            {
                let mut pass = pass!(enc, "bluestein_prepare", 2);
                pass.set_pipeline(&self.bluestein_prepare_pl.pipeline);
                pass.set_bind_group(0, &self.prepare_bg, &[]);
                let total = (self.n_beams * self.fft_m) as u32;
                pass.dispatch_workgroups((total + 255) / 256, 1, 1);
            }
            // 4b: Forward FFT 
            {
                let mut pass = pass!(enc, "fft_fwd", 3);
                pass.set_pipeline(&self.fft_twiddle_pl.pipeline);
                pass.set_bind_group(0, &self.fft_fwd_a_bg, &[]);
                pass.dispatch_workgroups(self.n_beams as u32, 1, 1);
            }
            // 4c: Pointwise multiply
            {
                let mut pass = pass!(enc, "pointwise_mul", 4);
                pass.set_pipeline(&self.pointwise_mul_pl.pipeline);
                pass.set_bind_group(0, &self.pointwise_mul_bg, &[]);
                let total = ((self.n_beams * self.fft_m) as u32 + 255) / 256;
                pass.dispatch_workgroups(total, 1, 1);
            }
            // 4d: Inverse FFT
            {
                let mut pass = pass!(enc, "fft_inv", 5);
                pass.set_pipeline(&self.fft_twiddle_pl.pipeline);
                pass.set_bind_group(0, &self.fft_inv_c_bg, &[]);
                pass.dispatch_workgroups(self.n_beams as u32, 1, 1);
            }
            // 4e: Finalize (chirp post-multiply + extract)
            {
                let mut pass = pass!(enc, "bluestein_finalize", 6);
                pass.set_pipeline(&self.bluestein_finalize_pl.pipeline);
                pass.set_bind_group(0, &self.finalize_bg, &[]);
                let total = ((self.n_beams * self.n_freq) as u32 + 255) / 256;
                pass.dispatch_workgroups(total, 1, 1);
            }
            timings.dft_us = duration_us(t_dft.elapsed()).max(1);
        }

        if let (Some(qs), Some(qr), Some(qst)) = (&self.query_set, &self.query_resolve_buf, &self.query_staging_buf) {
            enc.resolve_query_set(qs, 0..14, qr, 0);
            enc.copy_buffer_to_buffer(qr, 0, qst, 0, 14 * 8);
        }

        
        q.submit([enc.finish()]);

        timings.scatter_us  = duration_us(t_gpu.elapsed());

        // ============ READBACK ============a
        let t_readback = Instant::now();
        let beam_bytes = (self.n_beams * self.n_freq * size_of::<f32>()) as u64;
        {
            let mut enc = d.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(&self.beam_re_buf, 0, &self.staging_buf, 0,          beam_bytes);
            enc.copy_buffer_to_buffer(&self.beam_im_buf, 0, &self.staging_buf, beam_bytes, beam_bytes);
            q.submit([enc.finish()]);
        }
        let slice = self.staging_buf.slice(..);
        let q_slice = self.query_staging_buf.as_ref().map(|b| b.slice(..));

        let (tx, rx) = mpsc::channel();
        let (tx_q, rx_q) = mpsc::channel();

        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        if let Some(qs) = &q_slice {
            qs.map_async(wgpu::MapMode::Read, move |res| { let _ = tx_q.send(res); });
        }
        
        // spin-loop 
        let timeout_ms = parse_u64_env("SONAR_WGPU_MAP_TIMEOUT_MS", 5000);
        let timeout = Instant::now() + Duration::from_millis(timeout_ms);
        loop {
            d.poll(wgpu::Maintain::Poll);
            if let Ok(res) = rx.try_recv() {
                res.map_err(|e| format!("map_async failed: {e:?}"))?;
                break;
            }
            if Instant::now() > timeout {
                return Err(format!("map_async timed out after {} ms", timeout_ms));
            }
            std::hint::spin_loop();
        }

        // Read query data if available
        if let Some(qs) = q_slice {
            // Also wait for query map to finish since we just read the main buffer
            if let Ok(Ok(())) = rx_q.try_recv() {
                let data = qs.get_mapped_range();
                let times: &[u64] = bytemuck::cast_slice(&data);
                if times.len() == 14 {
                    let us = |idx: usize| -> u64 {
                        let start = times[idx * 2];
                        let end = times[idx * 2 + 1];
                        if start > 0 && end >= start {
                            ((end - start) as f32 * self.timestamp_period / 1000.0) as u64
                        } else { 1 }
                    };
                    timings.scatter_us = us(0);
                    timings.beam_sum_us = 1; // Tightly fused
                    timings.correction_us = us(1);
                    timings.dft_us = us(2) + us(3) + us(4) + us(5) + us(6);
                }
                drop(data);
                self.query_staging_buf.as_ref().unwrap().unmap();
            }
        }


        let data = slice.get_mapped_range();
        let floats: &[f32] = bytemuck::cast_slice(&data);
        let n = self.n_beams * self.n_freq;
        out_real.copy_from_slice(&floats[..n]);
        out_imag.copy_from_slice(&floats[n..2 * n]);
        drop(data);
        self.staging_buf.unmap();
        timings.readback_us = duration_us(t_readback.elapsed()).max(1);
        timings.total_us    = duration_us(total_start.elapsed()).max(1);
        Ok(timings)
    }
}

// ============ C ABI ============

#[no_mangle]
pub extern "C" fn sonar_wgpu_create(n_beams: i32, n_rays: i32, n_freq: i32, ray_skips: i32) -> *mut SonarWgpuContext {
    if n_beams <= 0 || n_rays <= 0 || n_freq <= 0 || ray_skips == 0 {
        eprintln!("[sonar_wgpu] Invalid dimensions");
        return std::ptr::null_mut();
    }
    match SonarWgpuContext::new(
        n_beams as usize, n_rays as usize, n_freq as usize,
        ray_skips.max(1) as usize, 1500.0, 1000.0, 1.0, 0.1, 1.0, 200.0,
    ) {
        Ok(ctx)  => Box::into_raw(Box::new(ctx)),
        Err(e)   => { eprintln!("[sonar_wgpu] create failed: {e}"); std::ptr::null_mut() }
    }
}

#[no_mangle]
pub extern "C" fn sonar_wgpu_destroy(ctx: *mut SonarWgpuContext) {
    if ctx.is_null() { return; }
    unsafe { drop(Box::from_raw(ctx)); }
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn sonar_wgpu_run(
    ctx: *mut SonarWgpuContext,
    depth: *const f32, normals: *const f32, reflectivity: *const f32,
    beamCorrector: *const f32,
    soundSpeed: f32, bandwidth: f32, sonarFreq: f32,
    sourceLevel: f32, attenuation: f32, areaScaler: f32, maxDistance: f32,
    beamCorrectorSum: f32, frameSeed: u32,
    out_real: *mut f32, out_imag: *mut f32,
) {
    sonar_wgpu_run_profiled(
        ctx, depth, normals, reflectivity, beamCorrector,
        soundSpeed, bandwidth, sonarFreq, sourceLevel, attenuation,
        areaScaler, maxDistance, beamCorrectorSum, frameSeed,
        out_real, out_imag, std::ptr::null_mut(),
    );
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
#[allow(non_snake_case)]
pub extern "C" fn sonar_wgpu_run_profiled(
    ctx: *mut SonarWgpuContext,
    depth: *const f32, normals: *const f32, reflectivity: *const f32,
    beamCorrector: *const f32,
    soundSpeed: f32, bandwidth: f32, sonarFreq: f32,
    sourceLevel: f32, attenuation: f32, areaScaler: f32, maxDistance: f32,
    beamCorrectorSum: f32, frameSeed: u32,
    out_real: *mut f32, out_imag: *mut f32,
    timings_out: *mut SonarWgpuRunTimings,
) {
    if ctx.is_null() || depth.is_null() || normals.is_null() || reflectivity.is_null()
        || beamCorrector.is_null() || out_real.is_null() || out_imag.is_null()
    {
        eprintln!("[sonar_wgpu] Null pointer passed");
        return;
    }
    let c = unsafe { &mut *ctx };
    let n = c.n_rays * c.n_beams;
    let o = c.n_beams * c.n_freq;

    let depth_s  = unsafe { std::slice::from_raw_parts(depth, n) };
    let norms_s  = unsafe { std::slice::from_raw_parts(normals, n * 3) };
    let refl_s   = unsafe { std::slice::from_raw_parts(reflectivity, n) };
    let bc_s     = unsafe { std::slice::from_raw_parts(beamCorrector, c.n_beams * c.n_beams) };
    let out_r    = unsafe { std::slice::from_raw_parts_mut(out_real, o) };
    let out_i    = unsafe { std::slice::from_raw_parts_mut(out_imag, o) };

    match c.run(depth_s, norms_s, refl_s, bc_s,
        soundSpeed, bandwidth, sonarFreq, sourceLevel, attenuation,
        areaScaler, maxDistance, beamCorrectorSum, frameSeed,
        out_r, out_i)
    {
        Ok(t) => { if !timings_out.is_null() { unsafe { *timings_out = t; } } }
        Err(e) => {
            eprintln!("[sonar_wgpu] run error: {e}");
            out_r.fill(0.0); out_i.fill(0.0);
            if !timings_out.is_null() { unsafe { *timings_out = SonarWgpuRunTimings::default(); } }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_sonar_pipeline_flat_seafloor() {
        let _guard = TEST_LOCK.lock().expect("lock poisoned");
        let (n_beams, n_rays, n_freq) = (8, 16, 16);
        let ctx_ptr = sonar_wgpu_create(n_beams as i32, n_rays as i32, n_freq as i32, 1);
        if ctx_ptr.is_null() { eprintln!("No GPU, skipping"); return; }

        let total = n_rays * n_beams;
        let depth: Vec<f32> = vec![10.0; total];
        let mut normals: Vec<f32> = vec![0.0; total * 3];
        for i in 0..total { normals[i * 3 + 2] = 1.0; }
        let reflectivity = vec![1.0f32; total];
        let mut bc = vec![0.0f32; n_beams * n_beams];
        for b in 0..n_beams { bc[b * n_beams + b] = 1.0; }

        let mut out_real = vec![0.0f32; n_beams * n_freq];
        let mut out_imag = vec![0.0f32; n_beams * n_freq];
        let mut timings  = SonarWgpuRunTimings::default();

        sonar_wgpu_run_profiled(
            ctx_ptr,
            depth.as_ptr(), normals.as_ptr(), reflectivity.as_ptr(), bc.as_ptr(),
            1500.0, 1000.0, 300_000.0, 220.0, 0.001, 0.0001, 1000.0, 1.0, 42u32,
            out_real.as_mut_ptr(), out_imag.as_mut_ptr(), &mut timings,
        );

        for (i, &v) in out_real.iter().enumerate() {
            assert!(v.is_finite(), "out_real[{i}] = {v} is NaN/inf");
        }
        for (i, &v) in out_imag.iter().enumerate() {
            assert!(v.is_finite(), "out_imag[{i}] = {v} is NaN/inf");
        }
        let any_nonzero = out_real.iter().zip(out_imag.iter())
            .any(|(&r, &im)| r.abs() > 1e-30 || im.abs() > 1e-30);
        assert!(any_nonzero, "All outputs are zero — acoustic model is broken");

        println!("[test] out_real[0] = {:.4e}", out_real[0]);
        println!("[test] total_us = {}", timings.total_us);
        sonar_wgpu_destroy(ctx_ptr);
    }

    #[test]
    fn test_context_create_destroy() {
        let _guard = TEST_LOCK.lock().expect("lock poisoned");
        let ctx = sonar_wgpu_create(4, 8, 8, 1);
        if ctx.is_null() { return; }
        sonar_wgpu_destroy(ctx);
    }
}
