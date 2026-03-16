// Dual matmul correction kernel, O(M*K*N) total work.

@group(0) @binding(0) var<storage, read>       A:      array<f32>;
@group(0) @binding(1) var<storage, read>       B_re:   array<f32>;
@group(0) @binding(2) var<storage, read>       B_im:   array<f32>;
@group(0) @binding(3) var<storage, read_write> C_re:   array<f32>;
@group(0) @binding(4) var<storage, read_write> C_im:   array<f32>;

struct MatConfig { M: u32, K: u32, N: u32, _pad: u32 }
@group(0) @binding(5) var<uniform> cfg: MatConfig;

var<workgroup> tile_A:    array<f32, 256>;
var<workgroup> tile_B_re: array<f32, 256>;
var<workgroup> tile_B_im: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;
    
    var sum_re = 0.0f;
    var sum_im = 0.0f;
    
    let num_tiles = (cfg.K + 15u) / 16u;
    
    for (var tile = 0u; tile < num_tiles; tile++) {
        let a_col = tile * 16u + local_col;
        if row < cfg.M && a_col < cfg.K {
            // CUDA-compatible beamCorrector flattening: A[col * M + row].
            tile_A[local_row * 16u + local_col] = A[a_col * cfg.M + row];
        } else {
            tile_A[local_row * 16u + local_col] = 0.0;
        }
        
        let b_row = tile * 16u + local_row;
        if b_row < cfg.K && col < cfg.N {
            // Input B uses CUDA-compatible freq-major layout.
            tile_B_re[local_row * 16u + local_col] = B_re[col * cfg.K + b_row];
            tile_B_im[local_row * 16u + local_col] = B_im[col * cfg.K + b_row];
        } else {
            tile_B_re[local_row * 16u + local_col] = 0.0;
            tile_B_im[local_row * 16u + local_col] = 0.0;
        }
        
        workgroupBarrier();
        
        for (var k = 0u; k < 16u; k++) {
            let a_val = tile_A[local_row * 16u + k];
            sum_re += a_val * tile_B_re[k * 16u + local_col];
            sum_im += a_val * tile_B_im[k * 16u + local_col];
        }
        
        workgroupBarrier();
    }
    
    if row < cfg.M && col < cfg.N {
        C_re[row * cfg.N + col] = sum_re;
        C_im[row * cfg.N + col] = sum_im;
    }
}
