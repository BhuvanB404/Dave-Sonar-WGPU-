// In-place FFT kernel with twiddle table, O(N log2 N) per beam.

@group(0) @binding(0) var<storage, read_write> re:         array<f32>;
@group(0) @binding(1) var<storage, read_write> im:         array<f32>;
@group(0) @binding(2) var<storage, read>       twiddle_re: array<f32>;
@group(0) @binding(3) var<storage, read>       twiddle_im: array<f32>;

struct FftWgConfig {
    n:       u32,
    nBeams:  u32,
    inverse: u32,
    _pad:    u32,
}
@group(0) @binding(4) var<uniform> cfg: FftWgConfig;

var<workgroup> sh_re: array<f32, 1024>;
var<workgroup> sh_im: array<f32, 1024>;

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)       wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let beam   = wgid.x;
    let t      = lid.x;
    let n      = cfg.n;
    let half_n = n / 2u;

    if beam >= cfg.nBeams { return; }

    let base = beam * n;
    let num_stages = u32(round(log2(f32(n))));

    var load_idx = t;
    while load_idx < n {
        let j = reverseBits(load_idx) >> (32u - num_stages);
        sh_re[j] = re[base + load_idx];
        sh_im[j] = im[base + load_idx];
        load_idx += 256u;
    }
    workgroupBarrier();

    // Inverse FFT uses conjugated twiddle via sign flip.
    let inv_sign = select(1.0f, -1.0f, cfg.inverse == 1u);

    for (var stage = 0u; stage < num_stages; stage++) {
        let half_size = 1u << stage;
        let full_size = half_size << 1u;

        var pair_idx = t;
        while pair_idx < half_n {
            let group = pair_idx / half_size;
            let pair  = pair_idx % half_size;

            let i = group * full_size + pair;
            let j = i + half_size;

            let tw_idx = pair * (n >> (stage + 1u));

            let tw_re = twiddle_re[tw_idx];
            let tw_im = twiddle_im[tw_idx] * inv_sign;

            let a_re = sh_re[i];
            let a_im = sh_im[i];
            let b_re = sh_re[j];
            let b_im = sh_im[j];

            let tb_re = b_re * tw_re - b_im * tw_im;
            let tb_im = b_re * tw_im + b_im * tw_re;

            sh_re[i] = a_re + tb_re;
            sh_im[i] = a_im + tb_im;
            sh_re[j] = a_re - tb_re;
            sh_im[j] = a_im - tb_im;

            pair_idx += 256u;
        }
        workgroupBarrier();
    }

    if cfg.inverse == 1u {
        let inv_n = 1.0f / f32(n);
        var scale_idx = t;
        while scale_idx < n {
            sh_re[scale_idx] *= inv_n;
            sh_im[scale_idx] *= inv_n;
            scale_idx += 256u;
        }
        workgroupBarrier();
    }

    var store_idx = t;
    while store_idx < n {
        re[base + store_idx] = sh_re[store_idx];
        im[base + store_idx] = sh_im[store_idx];
        store_idx += 256u;
    }
}
