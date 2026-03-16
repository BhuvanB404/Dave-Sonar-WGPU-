// Fused scatter+sum kernel, O((nRays/raySkips) + log2(256)) per (beam,freq).

@group(0) @binding(0) var<storage, read>       depth:        array<f32>;
@group(0) @binding(1) var<storage, read>       normals:      array<f32>;
@group(0) @binding(2) var<storage, read>       reflectivity: array<f32>;
@group(0) @binding(3) var<storage, read_write> beam_re:      array<f32>;
@group(0) @binding(4) var<storage, read_write> beam_im:      array<f32>;

struct Config {
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
@group(0) @binding(5) var<uniform> cfg: Config;

var<workgroup> wg_re: array<f32, 256>;
var<workgroup> wg_im: array<f32, 256>;

const TWO_PI: f32 = 6.2831853;

fn pcg32(seed: u32) -> u32 {
    var s = seed * 747796405u + 2891336453u;
    let word = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (word >> 22u) ^ word;
}

fn rand_f32(state: ptr<function, u32>) -> f32 {
    *state = pcg32(*state);
    return f32(*state) / 4294967296.0;
}

fn rand_normal(state: ptr<function, u32>) -> f32 {
    let u1 = rand_f32(state) + 1e-10;
    let u2 = rand_f32(state);
    return sqrt(-2.0 * log(u1)) * cos(TWO_PI * u2);
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(workgroup_id)       wgid: vec3<u32>,
    @builtin(local_invocation_id) lid:  vec3<u32>,
) {
    let freq = wgid.x;
    let beam = wgid.y;
    let t    = lid.x;

    if freq >= cfg.nFreq || beam >= cfg.nBeams { return; }

    let nRaysSkipped = cfg.nRays / cfg.raySkips;

    // CUDA parity: preserve odd/even nFreq frequency-bin mapping exactly.
    var freq_hz: f32;
    if (cfg.nFreq % 2u == 0u) {
        freq_hz = cfg.delta_f * (-f32(cfg.nFreq) + 2.0 * (f32(freq) + 1.0)) * 0.5;
    } else {
        freq_hz = cfg.delta_f * (-(f32(cfg.nFreq) - 1.0) + 2.0 * (f32(freq) + 1.0)) * 0.5;
    }
    let kw_factor = TWO_PI / cfg.soundSpeed;

    var acc_re = 0.0f;
    var acc_im = 0.0f;

    var active_ray = t;
    while active_ray < nRaysSkipped {
        let ray = active_ray * cfg.raySkips;
        if ray < cfg.nRays {
            let flat_idx   = ray * cfg.nBeams + beam;
            let normal_idx = ray * cfg.nBeams * 3u + beam * 3u;

            let r   = depth[flat_idx];
            let nz  = normals[normal_idx + 2u];
            let rf  = max(reflectivity[flat_idx], 0.0);

            let valid = (r > 0.0) && (r <= cfg.maxDistance) && (rf > 0.0);
            if valid {
                let incidence    = acos(clamp(nz, -1.0, 1.0));
                let lambert_sqrt = sqrt(rf) * cos(incidence);
                let r_s          = max(r, 0.01);
                let prop         = (1.0 / (r_s * r_s)) * exp(-2.0 * cfg.attenuation * r_s);
                let area_sqrt    = sqrt(r_s * cfg.areaScaler);

                var rng = cfg.frameSeed ^ (flat_idx * 2654435761u);
                let xi_z = rand_normal(&rng);
                let xi_y = rand_normal(&rng);

                let inv_sqrt2 = 0.70710678f;
                let scale     = inv_sqrt2 * cfg.sourceTerm * prop * lambert_sqrt * area_sqrt;
                let amp_re    = xi_z * scale;
                let amp_im    = xi_y * scale;

                let phase = 2.0 * kw_factor * freq_hz * r_s;
                let c = cos(phase);
                let s = sin(phase);

                acc_re += amp_re * c - amp_im * s;
                acc_im += amp_re * s + amp_im * c;
            }
        }
        active_ray += 256u;
    }

    wg_re[t] = acc_re;
    wg_im[t] = acc_im;
    workgroupBarrier();

    var stride = 128u;
    while stride > 0u {
        if t < stride {
            wg_re[t] += wg_re[t + stride];
            wg_im[t] += wg_im[t + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if t == 0u {
        // CUDA-compatible output layout: freq-major (freq * nBeams + beam).
        beam_re[freq * cfg.nBeams + beam] = wg_re[0];
        beam_im[freq * cfg.nBeams + beam] = wg_im[0];
    }
}
