#pragma once

#include <complex>
#include <valarray>

#include <opencv2/core.hpp>

namespace NpsGazeboSonar
{

typedef std::complex<float> Complex;
typedef std::valarray<Complex> CArray;
typedef std::valarray<CArray> CArray2D;

CArray2D sonar_calculation_wrapper(
  const cv::Mat & depth_image, const cv::Mat & normal_image, double _hPixelSize, double _vPixelSize,
  double _hFOV, double _vFOV, double _beam_azimuthAngleWidth, double _beam_elevationAngleWidth,
  double _ray_azimuthAngleWidth, float * _ray_elevationAngles, double _ray_elevationAngleWidth,
  double _soundSpeed, double _maxDistance, double _sourceLevel, int _nBeams, int _nRays,
  int _raySkips, double _sonarFreq, double _bandwidth, int _nFreq,
  const cv::Mat & reflectivity_image, double _attenuation, float * _window, float ** _beamCorrector,
  float _beamCorrectorSum, bool _debugFlag, bool _blazingFlag);

}  // namespace NpsGazeboSonar
