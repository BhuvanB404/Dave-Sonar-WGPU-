/*
 * WGPU-backed implementation of the DAVE sonar calculation wrapper.
 */

#include "sonar_calculation_wgpu.hh"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <mutex>
#include <vector>

#include "sonar_wgpu.h"

namespace NpsGazeboSonar
{
namespace
{
std::mutex gCtxMutex;
SonarWgpuContext * gCtx{nullptr};
int gCtxBeams{0};
int gCtxRays{0};
int gCtxFreq{0};
int gCtxRaySkips{0};

const float * MatDataFloat(const cv::Mat & mat, int expectedType)
{
  if (mat.type() != expectedType) {
    return nullptr;
  }
  return mat.ptr<float>(0);
}

bool EnsureContext(int nBeams, int nRays, int nFreq, int raySkips)
{
  if (gCtx != nullptr &&
    gCtxBeams == nBeams &&
    gCtxRays == nRays &&
    gCtxFreq == nFreq &&
    gCtxRaySkips == raySkips)
  {
    return true;
  }

  if (gCtx != nullptr) {
    sonar_wgpu_destroy(gCtx);
    gCtx = nullptr;
  }

  gCtx = sonar_wgpu_create(nBeams, nRays, nFreq, raySkips);
  if (gCtx == nullptr) {
    return false;
  }

  gCtxBeams = nBeams;
  gCtxRays = nRays;
  gCtxFreq = nFreq;
  gCtxRaySkips = raySkips;
  return true;
}
}  // namespace

CArray2D sonar_calculation_wrapper(
  const cv::Mat & depth_image, const cv::Mat & normal_image, double /*_hPixelSize*/,
  double /*_vPixelSize*/, double /*_hFOV*/, double /*_vFOV*/, double /*_beam_azimuthAngleWidth*/,
  double /*_beam_elevationAngleWidth*/, double _ray_azimuthAngleWidth, float * /*_ray_elevationAngles*/,
  double _ray_elevationAngleWidth, double _soundSpeed, double _maxDistance, double _sourceLevel,
  int _nBeams, int _nRays, int _raySkips, double _sonarFreq, double _bandwidth, int _nFreq,
  const cv::Mat & reflectivity_image, double _attenuation, float * /*_window*/, float ** _beamCorrector,
  float _beamCorrectorSum, bool _debugFlag, bool _blazingFlag)
{
  CArray2D out(static_cast<size_t>(_nBeams));
  for (int b = 0; b < _nBeams; ++b) {
    out[static_cast<size_t>(b)] = CArray(static_cast<size_t>(_nFreq));
    for (int f = 0; f < _nFreq; ++f) {
      out[static_cast<size_t>(b)][static_cast<size_t>(f)] = Complex(0.0f, 0.0f);
    }
  }

  if (_nBeams <= 0 || _nRays <= 0 || _nFreq <= 0) {
    return out;
  }

  const float * depthPtr = MatDataFloat(depth_image, CV_32FC1);
  const float * normalPtr = MatDataFloat(normal_image, CV_32FC3);
  const float * reflectPtr = MatDataFloat(reflectivity_image, CV_32FC1);
  if (depthPtr == nullptr || normalPtr == nullptr || reflectPtr == nullptr) {
    return out;
  }

    std::vector<float> beamCorrectorLin(static_cast<size_t>(_nBeams) * static_cast<size_t>(_nBeams), 0.0f);
    for (int beam = 0; beam < _nBeams; ++beam) {
      for (int beamOther = 0; beamOther < _nBeams; ++beamOther) {
        // Match CUDA flattening exactly:
        // beamCorrector_lin_h[beam_other * nBeams + beam] = beamCorrector[beam][beam_other]
        beamCorrectorLin[
          static_cast<size_t>(beamOther) * static_cast<size_t>(_nBeams) + static_cast<size_t>(beam)] =
          _beamCorrector[beam][beamOther];
      }
    }

  std::vector<float> outReal(static_cast<size_t>(_nBeams) * static_cast<size_t>(_nFreq), 0.0f);
  std::vector<float> outImag(static_cast<size_t>(_nBeams) * static_cast<size_t>(_nFreq), 0.0f);

  const float areaScaler = static_cast<float>(_ray_azimuthAngleWidth * _ray_elevationAngleWidth);
  const int raySkips = std::max(1, _raySkips);

  {
    std::lock_guard<std::mutex> lock(gCtxMutex);
    if (!EnsureContext(_nBeams, _nRays, _nFreq, raySkips)) {
      return out;
    }

    uint32_t frameSeed = _blazingFlag ? static_cast<uint32_t>(std::time(nullptr)) : 1234u;
    SonarWgpuRunTimings timings{};
    sonar_wgpu_run_profiled(
      gCtx,
      depthPtr,
      normalPtr,
      reflectPtr,
      beamCorrectorLin.data(),
      static_cast<float>(_soundSpeed),
      static_cast<float>(_bandwidth),
      static_cast<float>(_sonarFreq),
      static_cast<float>(_sourceLevel),
      static_cast<float>(_attenuation),
      areaScaler,
      static_cast<float>(_maxDistance),
      _beamCorrectorSum,
      frameSeed,
      outReal.data(),
      outImag.data(),
      &timings);

    if (_debugFlag) {
      std::cerr << "[sonar_wgpu] timings_us"
                << " upload=" << timings.upload_us
                << " scatter=" << timings.scatter_us
                << " beam_sum=" << timings.beam_sum_us
                << " readback=" << timings.readback_us
                << " fallback_scatter_sum=" << timings.fallback_scatter_sum_us
                << " correction=" << timings.correction_us
                << " dft=" << timings.dft_us
                << " total=" << timings.total_us
                << std::endl;
    }
  }

  for (int beam = 0; beam < _nBeams; ++beam) {
    for (int f = 0; f < _nFreq; ++f) {
      const size_t idx = static_cast<size_t>(beam) * static_cast<size_t>(_nFreq) + static_cast<size_t>(f);
      out[static_cast<size_t>(beam)][static_cast<size_t>(f)] = Complex(outReal[idx], outImag[idx]);
    }
  }

  return out;
}

}  // namespace NpsGazeboSonar
