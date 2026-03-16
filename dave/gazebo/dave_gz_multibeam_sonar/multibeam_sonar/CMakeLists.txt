cmake_minimum_required(VERSION 3.11.0 FATAL_ERROR)

set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL
    "Suppress developer warnings" FORCE)
cmake_policy(SET CMP0144 NEW)

project(multibeam_sonar)

set(MULTIBEAM_BACKEND "WGPU" CACHE STRING "Backend to use: CUDA, WGPU, or SYCL")
set_property(CACHE MULTIBEAM_BACKEND PROPERTY STRINGS CUDA WGPU SYCL)
option(MULTIBEAM_USE_WGPU "Deprecated. Use MULTIBEAM_BACKEND=WGPU" ON)
set(CUDA_ARCHITECTURE "60" CACHE STRING "Target CUDA SM version")
set(SONAR_WGPU_ROOT "/home/bb/sonar-gpu/implementation/sonar_wgpu" CACHE PATH
    "Path to sonar_wgpu Rust crate root")
set(SONAR_SYCL_ROOT "/home/bb/sonar-gpu/implementation/sonar_sycl" CACHE PATH
    "Path to sonar_sycl C++ project root")

string(TOUPPER "${MULTIBEAM_BACKEND}" MULTIBEAM_BACKEND)

find_package(ament_cmake REQUIRED)
find_package(gz-cmake3 REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-sensors8 REQUIRED)
find_package(gz-rendering8 REQUIRED OPTIONAL_COMPONENTS ogre ogre2)
find_package(gz-transport13 REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(OpenCV REQUIRED)
find_package(marine_acoustic_msgs REQUIRED)
find_package(cv_bridge REQUIRED)

set(GZ_MSGS_VER ${gz-msgs10_VERSION_MAJOR})
set(GZ_RENDERING_VER ${gz-rendering8_VERSION_MAJOR})
set(GZ_SENSORS_VER ${gz-sensors8_VERSION_MAJOR})
set(GZ_TRANSPORT_VER ${gz-transport13_VERSION_MAJOR})
set(GZ_SIM_VER ${gz-sim8_VERSION_MAJOR})

if(TARGET gz-rendering${GZ_RENDERING_VER}::ogre)
  set(HAVE_OGRE TRUE)
  set(GZ_RENDERING_TARGET gz-rendering${GZ_RENDERING_VER}-ogre)
  add_definitions(-DWITH_OGRE)
endif()

if(TARGET gz-rendering${GZ_RENDERING_VER}::ogre2)
  set(HAVE_OGRE2 TRUE)
  set(GZ_RENDERING_TARGET gz-rendering${GZ_RENDERING_VER}-ogre2)
  add_definitions(-DWITH_OGRE2)
endif()

if(MULTIBEAM_BACKEND STREQUAL "WGPU")
  set(SONAR_WGPU_INCLUDE_DIR "${SONAR_WGPU_ROOT}/include")
  find_library(SONAR_WGPU_LIB
    NAMES sonar_wgpu
    PATHS "${SONAR_WGPU_ROOT}/target/release"
    NO_DEFAULT_PATH
  )
  if(NOT SONAR_WGPU_LIB)
    message(FATAL_ERROR "sonar_wgpu library not found. Build ${SONAR_WGPU_ROOT} first.")
  endif()

  add_library(${PROJECT_NAME}
    SHARED
      MultibeamSonarSensor.cc
      sonar_calculation_wgpu.cc
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE DAVE_USE_WGPU=1)

  target_include_directories(${PROJECT_NAME}
    PUBLIC
      ${gz-sensors${GZ_SENSORS_VER}_INCLUDE_DIRS}
      ${gz-rendering${GZ_RENDERING_VER}_INCLUDE_DIRS}
      ${gz-msgs${GZ_MSGS_VER}_INCLUDE_DIRS}
      ${gz-transport${GZ_TRANSPORT_VER}_INCLUDE_DIRS}
      ${gz-sim${GZ_SIM_VER}_INCLUDE_DIRS}
      ${SONAR_WGPU_INCLUDE_DIR}
  )

  target_link_libraries(${PROJECT_NAME}
    ${SONAR_WGPU_LIB}
  )

  set_target_properties(${PROJECT_NAME}
    PROPERTIES
      BUILD_RPATH "${SONAR_WGPU_ROOT}/target/release"
      INSTALL_RPATH "${SONAR_WGPU_ROOT}/target/release"
  )
elseif(MULTIBEAM_BACKEND STREQUAL "SYCL")
  set(SONAR_SYCL_INCLUDE_DIR "${SONAR_SYCL_ROOT}/include")
  find_library(SONAR_SYCL_LIB
    NAMES sonar_sycl
    PATHS "${SONAR_SYCL_ROOT}/build"
    NO_DEFAULT_PATH
  )
  if(NOT SONAR_SYCL_LIB)
    message(FATAL_ERROR "sonar_sycl library not found. Build ${SONAR_SYCL_ROOT} first.")
  endif()

  add_library(${PROJECT_NAME}
    SHARED
      MultibeamSonarSensor.cc
      sonar_calculation_sycl.cc
  )
  target_compile_definitions(${PROJECT_NAME} PRIVATE DAVE_USE_SYCL=1)

  target_include_directories(${PROJECT_NAME}
    PUBLIC
      ${gz-sensors${GZ_SENSORS_VER}_INCLUDE_DIRS}
      ${gz-rendering${GZ_RENDERING_VER}_INCLUDE_DIRS}
      ${gz-msgs${GZ_MSGS_VER}_INCLUDE_DIRS}
      ${gz-transport${GZ_TRANSPORT_VER}_INCLUDE_DIRS}
      ${gz-sim${GZ_SIM_VER}_INCLUDE_DIRS}
      ${SONAR_SYCL_INCLUDE_DIR}
  )

  target_link_libraries(${PROJECT_NAME}
    ${SONAR_SYCL_LIB}
  )

  set_target_properties(${PROJECT_NAME}
    PROPERTIES
      BUILD_RPATH "${SONAR_SYCL_ROOT}/build"
      INSTALL_RPATH "${SONAR_SYCL_ROOT}/build"
  )
elseif(MULTIBEAM_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  message(STATUS "CUDA found, enabling CUDA support.")
  include_directories(${CUDAToolkit_INCLUDE_DIRS})
  set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURE})

  add_library(${PROJECT_NAME}
    SHARED
      MultibeamSonarSensor.cc
      sonar_calculation_cuda.cu
  )

  set_target_properties(${PROJECT_NAME}
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
  )

  target_include_directories(${PROJECT_NAME}
    PUBLIC
      ${gz-sensors${GZ_SENSORS_VER}_INCLUDE_DIRS}
      ${gz-rendering${GZ_RENDERING_VER}_INCLUDE_DIRS}
      ${gz-msgs${GZ_MSGS_VER}_INCLUDE_DIRS}
      ${CUDAToolkit_INCLUDE_DIRS}
      ${gz-transport${GZ_TRANSPORT_VER}_INCLUDE_DIRS}
      ${gz-sim${GZ_SIM_VER}_INCLUDE_DIRS}
  )

  set(CUDA_RT_LIB "")
  set(CUDA_FFT_LIB "")
  set(CUDA_BLAS_LIB "")

  if(TARGET CUDAToolkit::cudart)
    set(CUDA_RT_LIB CUDAToolkit::cudart)
  elseif(TARGET CUDA::cudart)
    set(CUDA_RT_LIB CUDA::cudart)
  else()
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_RT_LIB ${CUDA_LIBRARIES})
  endif()

  if(TARGET CUDAToolkit::cufft)
    set(CUDA_FFT_LIB CUDAToolkit::cufft)
  elseif(TARGET CUDA::cufft)
    set(CUDA_FFT_LIB CUDA::cufft)
  else()
    set(CUDA_FFT_LIB ${CUDA_CUFFT_LIBRARIES})
  endif()

  if(TARGET CUDAToolkit::cublas)
    set(CUDA_BLAS_LIB CUDAToolkit::cublas)
  elseif(TARGET CUDA::cublas)
    set(CUDA_BLAS_LIB CUDA::cublas)
  else()
    find_library(CUBLAS_LIB cublas
      HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        /usr/local/cuda/lib64
        /usr/lib/x86_64-linux-gnu
        /usr/lib
    )
    if(NOT CUBLAS_LIB)
      message(FATAL_ERROR "cuBLAS not found")
    endif()
    set(CUDA_BLAS_LIB ${CUBLAS_LIB})
  endif()

  target_link_libraries(${PROJECT_NAME}
    ${CUDA_RT_LIB}
    ${CUDA_FFT_LIB}
    ${CUDA_BLAS_LIB}
  )
else()
  message(FATAL_ERROR "Invalid MULTIBEAM_BACKEND=${MULTIBEAM_BACKEND}. Use CUDA, WGPU, or SYCL")
endif()

install(TARGETS ${PROJECT_NAME}
  DESTINATION lib/${PROJECT_NAME}
)

ament_target_dependencies(${PROJECT_NAME}
  gz-sensors${GZ_SENSORS_VER}
  ${GZ_RENDERING_TARGET}
  gz-msgs${GZ_MSGS_VER}
  gz-transport${GZ_TRANSPORT_VER}
  gz-sim${GZ_SIM_VER}
  rclcpp
  sensor_msgs
  rosidl_default_generators
  OpenCV
  marine_acoustic_msgs
  geometry_msgs
  cv_bridge
)

ament_environment_hooks(
  "${CMAKE_CURRENT_SOURCE_DIR}/hooks/${PROJECT_NAME}.dsv.in"
)

ament_package()

