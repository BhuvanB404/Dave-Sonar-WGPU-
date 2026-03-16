#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Opaque GPU context. */
typedef struct SonarWgpuContext SonarWgpuContext;

/** Run timing values in microseconds. */
typedef struct SonarWgpuRunTimings
{
  uint64_t upload_us;
  uint64_t scatter_us;
  uint64_t beam_sum_us;
  uint64_t readback_us;
  uint64_t fallback_scatter_sum_us;
  uint64_t correction_us;
  uint64_t dft_us;
  uint64_t total_us;
} SonarWgpuRunTimings;

/** Create GPU context. */
SonarWgpuContext* sonar_wgpu_create(int n_beams, int n_rays,
                                    int n_freq, int ray_skips);

/** Destroy GPU context. */
void sonar_wgpu_destroy(SonarWgpuContext* ctx);

/** Run one sonar frame. */
void sonar_wgpu_run(
    SonarWgpuContext* ctx,
  const float*      depth,
    const float*      normals,
    const float*      reflectivity,
    const float*      beam_corrector,
    float             sound_speed,
    float             bandwidth,
    float             sonar_freq,
    float             source_level,
    float             attenuation,     
    float             area_scaler,     
    float             max_distance,    
    float             beam_corrector_sum, 
    uint32_t          frame_seed,      
    float*            out_real,
    float*            out_imag
);

/** Run one sonar frame and return timings. */
void sonar_wgpu_run_profiled(
    SonarWgpuContext* ctx,
    const float*      depth,
    const float*      normals,
    const float*      reflectivity,
    const float*      beam_corrector,
    float             sound_speed,
    float             bandwidth,
    float             sonar_freq,
    float             source_level,
    float             attenuation,
    float             area_scaler,
    float             max_distance,
    float             beam_corrector_sum,
    uint32_t          frame_seed,
    float*            out_real,
    float*            out_imag,
    SonarWgpuRunTimings* timings_out
);

#ifdef __cplusplus
}  /* extern "C" */
#endif
