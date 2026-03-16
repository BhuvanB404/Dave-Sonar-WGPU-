// Point-wise complex multiply kernel, O(nBeams * m).

@group(0) @binding(0) var<storage, read> a_re: array<f32>;
@group(0) @binding(1) var<storage, read> a_im: array<f32>;
@group(0) @binding(2) var<storage, read> b_re: array<f32>;
@group(0) @binding(3) var<storage, read> b_im: array<f32>;
@group(0) @binding(4) var<storage, read_write> c_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> c_im: array<f32>;

struct MulConfig {
    n: u32,
    m: u32,
    _pad0: u32,
    _pad1: u32,
}
@group(0) @binding(6) var<uniform> cfg: MulConfig;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= cfg.n { return; }

    let a_r = a_re[i];
    let a_i = a_im[i];
    
    let b_idx = i % cfg.m;
    let b_r = b_re[b_idx];
    let b_i = b_im[b_idx];

    c_re[i] = a_r * b_r - a_i * b_i;
    c_im[i] = a_r * b_i + a_i * b_r;
}
