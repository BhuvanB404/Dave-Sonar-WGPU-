// Bluestein prepare kernel, O(nBeams * m).

@group(0) @binding(0) var<storage, read> cor_re: array<f32>;
@group(0) @binding(1) var<storage, read> cor_im: array<f32>;
@group(0) @binding(2) var<storage, read> chirp_conj_re: array<f32>;
@group(0) @binding(3) var<storage, read> chirp_conj_im: array<f32>;
@group(0) @binding(4) var<storage, read_write> a_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> a_im: array<f32>;

struct PrepareConfig {
    nBeams: u32,
    nFreq: u32,
    m: u32,
    _pad: u32,
}
@group(0) @binding(6) var<uniform> cfg: PrepareConfig;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_idx = gid.x;
    let total = cfg.nBeams * cfg.m;
    if global_idx >= total { return; }

    let beam = global_idx / cfg.m;
    let idx = global_idx % cfg.m;

    let out_offset = beam * cfg.m + idx;

    if idx < cfg.nFreq {
        let in_offset = beam * cfg.nFreq + idx;
        let x_re = cor_re[in_offset];
        let x_im = cor_im[in_offset];
        let c_re = chirp_conj_re[idx];
        let c_im = chirp_conj_im[idx];

        a_re[out_offset] = x_re * c_re - x_im * c_im;
        a_im[out_offset] = x_re * c_im + x_im * c_re;
    } else {
        a_re[out_offset] = 0.0;
        a_im[out_offset] = 0.0;
    }
}
