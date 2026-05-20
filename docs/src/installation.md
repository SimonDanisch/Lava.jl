# Installation

## Julia package

Lava is not yet in the General registry. Install directly from the repository:

```julia
import Pkg
Pkg.add(url="https://github.com/SimonDanisch/Lava.jl")
```

Once tagged it will be installable via `Pkg.add("Lava")`.

## System requirements

* **Julia 1.11 or newer**
* A **Vulkan 1.2+ driver** with the `bufferDeviceAddress` and `variablePointers` features. Any recent NVIDIA, AMD, Intel, MoltenVK (macOS), or lavapipe ≥ 24.x driver qualifies.
* For ray tracing: a driver exposing `VK_KHR_ray_tracing_pipeline` and `VK_KHR_acceleration_structure`. Software RT works via lavapipe's emulation, but RADV / NVIDIA give real hardware BVH traversal.
* For graphics: an X11 or Wayland display server, and GLFW build dependencies — on Debian-likes:
  ```
  apt-get install xorg-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev mesa-utils
  ```

Headless servers without `DISPLAY` can run Lava under `xvfb-run`:

```
DISPLAY=:0 xvfb-run -s '-screen 0 1024x768x24' julia --project ...
```

This is exactly how the CI runs the test suite.

## Verify the install

```julia
using Lava

ctx = Lava.vk_context()
@show ctx.device_name
@show ctx.rt_pipeline_properties !== nothing
```

A successful run prints the chosen GPU and whether hardware ray tracing is available. Anything else (driver missing, surface-extension errors, validation messages) points to a system-level issue — see [Debugging](debugging.md).

## Picking a device

If multiple Vulkan devices are present, Lava picks the first discrete GPU by default. To override:

```julia
ENV["LAVA_FORCE_DEVICE"] = "lavapipe"   # substring match against device name
using Lava
```

Common values: `"NVIDIA"`, `"RADV"`, `"Intel"`, `"llvmpipe"`, `"lavapipe"`, `"MoltenVK"`.
