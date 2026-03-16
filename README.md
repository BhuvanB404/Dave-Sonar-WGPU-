# WGPU Sonar 

This document summarizes the WGPU implementation layout, key file changes, API updates, and minimal Ubuntu host commands to run a world example.

## 1. Scope

This POC adds a WGPU backend path for the DAVE multibeam sonar plugin and connects it to the Rust library in `sonar_wgpu`.

## 2. WGPU Structure and File Map

### 2.1 Build and backend switch
- `dave/gazebo/dave_gz_multibeam_sonar/multibeam_sonar/CMakeLists.txt`
  - Selects backend with `MULTIBEAM_BACKEND` (`CUDA`, `WGPU`)
  - In WGPU mode, compiles `sonar_calculation_wgpu.cc`
  - Links `libsonar_wgpu.so` from `sonar_wgpu/target/release`

### 2.2 Plugin layer
- `dave/gazebo/dave_gz_multibeam_sonar/multibeam_sonar/MultibeamSonarSensor.cc`
  - Uses WGPU wrapper when WGPU backend is enabled
- `dave/gazebo/dave_gz_multibeam_sonar/multibeam_sonar/sonar_calculation_wgpu.hh`
  - WGPU wrapper interface used by the plugin
- `dave/gazebo/dave_gz_multibeam_sonar/multibeam_sonar/sonar_calculation_wgpu.cc`
  - Calls C API from `sonar_wgpu` and marshals plugin data

### 2.3 Rust API layer
- `sonar_wgpu/include/sonar_wgpu.h`
  - C ABI used by C++ plugin
- `sonar_wgpu/src/lib.rs`
  - Rust implementation of the sonar pipeline
- `sonar_wgpu/src/shaders/*.wgsl`
  - WGSL compute shaders used by the Rust pipeline

## 3. API Changes (C ABI)

File: `sonar_wgpu/include/sonar_wgpu.h`

### 3.1 New/used API surface
- `SonarWgpuContext* sonar_wgpu_create(...)`
- `void sonar_wgpu_destroy(...)`
- `void sonar_wgpu_run(...)`
- `void sonar_wgpu_run_profiled(...)`
- `SonarWgpuRunTimings` timing struct

### 3.2 Behavioral/API differences from older CUDA-style flow
- `rand_image` input is removed from WGPU API
- `frame_seed` is used for per-frame noise generation
- Optional stage timings are returned through `sonar_wgpu_run_profiled`

## 4. Important Notes About Tracked Files

- The test run is in the wgpu_test_run folder with the csv and generated sonar image.

## 5. Ubuntu Prerequisites

Host OS:
- Ubuntu 24.04 (recommended)

Required packages:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake pkg-config git curl ca-certificates \
  python3 python3-pip python3-venv \
  vulkan-tools libvulkan1 mesa-vulkan-drivers
```

Install Rust toolchain:

```bash
curl https://sh.rustup.rs -sSf | sh -s -- -y
source "$HOME/.cargo/env"
```

Install ROS 2 Jazzy and colcon:

```bash
sudo apt-get install -y software-properties-common
sudo add-apt-repository universe -y
sudo apt-get update
sudo apt-get install -y ros-jazzy-desktop python3-colcon-common-extensions
```

GPU prerequisites:
- Install your vendor GPU driver
- Verify Vulkan is available:

```bash
vulkaninfo --summary
```

## 6. Build and Run World Example (WGPU)

Use this style for reproducible local runs:
- parameterize paths once
- keep all generated outputs in one run directory
- do not hardcode user-specific paths

Set shared variables first:

```bash
export REPO_ROOT="$(pwd)"
export IMPL_ROOT="$REPO_ROOT/implementation"
export SONAR_WGPU_ROOT="$IMPL_ROOT/sonar_wgpu"
export DAVE_WS_ROOT="$IMPL_ROOT/dave_ws"
export RUN_DIR="$REPO_ROOT/artifacts/world_runs/wgpu_example"
```

### 6.1 Build Rust WGPU library

```bash
set -euo pipefail
source "$HOME/.cargo/env"
cd "$SONAR_WGPU_ROOT"
cargo build --release
```

### 6.2 Build DAVE sonar packages with WGPU backend

```bash
source /opt/ros/jazzy/setup.bash
cd "$DAVE_WS_ROOT"
colcon build --symlink-install \
  --packages-select multibeam_sonar multibeam_sonar_system \
  --cmake-clean-cache \
  --cmake-args \
    -DMULTIBEAM_BACKEND=WGPU \
    -DSONAR_WGPU_ROOT="$SONAR_WGPU_ROOT"
```

### 6.3 Run example world

```bash
mkdir -p "$RUN_DIR"

cd "$RUN_DIR"
find . -maxdepth 1 -type f \( \
  -name 'SonarRawData_*.csv' -o \
  -name 'SonarRawData_beam_angles.csv' -o \
  -name 'debug_timings.txt' -o \
  -name 'launch.log' \
\) -delete

set -euo pipefail
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH PYTHONPATH
source /opt/ros/jazzy/setup.bash
source "$DAVE_WS_ROOT/install/setup.bash"
export LD_LIBRARY_PATH="$SONAR_WGPU_ROOT/target/release:${LD_LIBRARY_PATH:-}"
export SONAR_WGPU_BACKENDS=vulkan
export SONAR_WGPU_PREFER_NVIDIA=1
cd "$RUN_DIR"

timeout 35s ros2 launch dave_demos dave_sensor.launch.py \
  namespace:=blueview_p900 \
  world_name:=dave_multibeam_sonar \
  paused:=false gui:=true headless:=true debug:=true verbosity_level:=3 \
  x:=0 y:=0 z:=2.0 roll:=0 pitch:=0 yaw:=0
```

### 6.4 Verify run output

```bash
cd "$RUN_DIR"

ls SonarRawData_*.csv
ls SonarRawData_beam_angles.csv
```

Expected:
- CSV count greater than 0
- Beam angles file present
- Adapter line in log (NVIDIA/Vulkan)

## 7. Optional Plot Generation

```bash
cd "$IMPL_ROOT"
python3 generate_plots.py \
  --base "$RUN_DIR" \
  --prefix wgpu_example
```

Outputs are written into the same `wgpu_example` folder.

## 8.  Distrobox Commands

 This  was tested and run using distrobox.

If you want to use the same tested container flow, use:

```bash
export REPO_ROOT="$(pwd)"
export IMPL_ROOT="$REPO_ROOT/implementation"
export SONAR_WGPU_ROOT="$IMPL_ROOT/sonar_wgpu"
export DAVE_WS_ROOT="$IMPL_ROOT/dave_ws"
export RUN_DIR="$REPO_ROOT/artifacts/world_runs/wgpu_example"
export CONTAINER_NAME="gz-jellico-nvidia"

# Build Rust WGPU library
distrobox enter "$CONTAINER_NAME" -- bash -lc '
set -euo pipefail
source "$HOME/.cargo/env"
cd "$SONAR_WGPU_ROOT"
cargo build --release
'

# Build DAVE packages with WGPU backend
distrobox enter "$CONTAINER_NAME" -- bash -lc '
set -euo pipefail
source /opt/ros/jazzy/setup.bash
cd "$DAVE_WS_ROOT"
colcon build --symlink-install \
  --packages-select multibeam_sonar multibeam_sonar_system \
  --cmake-clean-cache \
  --cmake-args \
    -DMULTIBEAM_BACKEND=WGPU \
    -DSONAR_WGPU_ROOT="$SONAR_WGPU_ROOT"
'

# Run world example
mkdir -p "$RUN_DIR"
cd "$RUN_DIR"
find . -maxdepth 1 -type f \( \
  -name "SonarRawData_*.csv" -o \
  -name "SonarRawData_beam_angles.csv" -o \
  -name "debug_timings.txt" -o \
  -name "launch.log" \
\) -delete

distrobox enter "$CONTAINER_NAME" -- bash -lc '
set -euo pipefail
unset AMENT_PREFIX_PATH CMAKE_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH PYTHONPATH
source /opt/ros/jazzy/setup.bash
source "$DAVE_WS_ROOT/install/setup.bash"
export LD_LIBRARY_PATH="$SONAR_WGPU_ROOT/target/release:${LD_LIBRARY_PATH:-}"
export SONAR_WGPU_BACKENDS=vulkan
export SONAR_WGPU_PREFER_NVIDIA=1
cd "$RUN_DIR"

timeout 35s ros2 launch dave_demos dave_sensor.launch.py \
  namespace:=blueview_p900 \
  world_name:=dave_multibeam_sonar \
  paused:=false gui:=true headless:=true debug:=true verbosity_level:=3 \
  x:=0 y:=0 z:=2.0 roll:=0 pitch:=0 yaw:=0
'
```
