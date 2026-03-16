// Bluestein finalize kernel, O(nBeams * nFreq).

@group(0) @binding(0) var<storage, read> c_re: array<f32>;
@group(0) @binding(1) var<storage, read> c_im: array<f32>;
@group(0) @binding(2) var<storage, read> chirp_conj_re: array<f32>;
@group(0) @binding(3) var<storage, read> chirp_conj_im: array<f32>;
@group(0) @binding(4) var<storage, read_write> beam_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> beam_im: array<f32>;

struct FinalizeConfig {
    nBeams: u32,
    nFreq: u32,
    m: u32,
    delta_f: f32,
}
@group(0) @binding(6) var<uniform> cfg: FinalizeConfig;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let global_idx = gid.x;
    let total = cfg.nBeams * cfg.nFreq;
    if global_idx >= total { return; }

    let beam = global_idx / cfg.nFreq;
    let k = global_idx % cfg.nFreq;

    let c_offset = beam * cfg.m + k;
    let c_r = c_re[c_offset];
    let c_i = c_im[c_offset];

    let chirp_r = chirp_conj_re[k];
    let chirp_i = chirp_conj_im[k];

    let out_offset = beam * cfg.nFreq + k;
    beam_re[out_offset] = (c_r * chirp_r - c_i * chirp_i) * cfg.delta_f;
    beam_im[out_offset] = (c_r * chirp_i + c_i * chirp_r) * cfg.delta_f;
}
