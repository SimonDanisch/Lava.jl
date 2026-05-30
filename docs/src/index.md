```@meta
CurrentModule = Lava
```

# Lava.jl

A Julia GPU backend that compiles Julia code to SPIR-V, targeting Vulkan.

Lava is three backends in one package, all sharing a single buffer model:

* **Compute** — full [GPUArrays.jl](https://github.com/JuliaGPU/GPUArrays.jl) and [KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) interface. Set `device = LavaBackend()` and any KA-based GPU code runs on Vulkan.
* **Graphics** — vertex, fragment, geometry, and tessellation shaders written in plain Julia (not GLSL), driven through high-level pipelines and `RenderWindow`s.
* **Ray tracing** — `VK_KHR_ray_tracing_pipeline` with raygen / closest-hit / any-hit / miss / intersection shaders as Julia functions and a Julia-side TLAS that supports incremental updates.

A single `LavaArray{T,N}` is usable across all stages: compute kernels can fill geometry, vertex shaders can render it, ray tracing shaders can trace against it — no copies, no format conversion.

## Where to start

| If you want to… | Go to |
|---|---|
| Install and verify your setup | [Installation](installation.md) |
| Use `LavaBackend()` like `CUDABackend()` / `ROCBackend()` | [Compute](compute.md) |
| Write vertex/fragment shaders in Julia | [Graphics](graphics.md) |
| Trace rays against a TLAS | [Ray Tracing](raytracing.md) |
| Tune dispatch latency, memory transfer, scheduling | [Performance](performance.md) |
| Understand the LLVM-IR → SPIR-V pipeline | [Architecture](architecture.md) |
| Diagnose a device-lost / validation error | [Debugging](debugging.md) |
| Check what's known broken | [Known Issues](known_issues.md) |
| Look up a specific function | [API Reference](api.md) |

## Status

The KernelAbstractions / GPUArrays compute path is feature-complete and well-tested (~99.7 % of the GPUArrays test suite passes). The graphics and ray tracing paths are functional and stable enough that the [Hikari](https://github.com/SimonDanisch/Hikari.jl) volumetric path tracer renders end-to-end through both, but their public APIs are still evolving — expect minor breaking changes between 0.x releases.

Performance figures on AMD RX 7900 XTX (vs AMDGPU.jl and pbrt-v4) live in the README and the [Performance](performance.md) page.

## Hardware

Tested on:

* AMD RX 7900 XTX (Vulkan 1.4, RADV)
* AMD Radeon 8060S (Vulkan 1.4, RADV)
* NVIDIA RTX 3070 Laptop (Vulkan 1.3)
* lavapipe (software, used for CI)

Requires Vulkan 1.2+ with `bufferDeviceAddress` and `variablePointers`. macOS is supported via MoltenVK.
