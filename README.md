# Lava.jl

A Julia GPU backend that compiles Julia code to SPIR-V, targeting Vulkan. Lava.jl is three things at once:

1. **A compute GPU backend** with full GPUArrays.jl + KernelAbstractions.jl interface. Set `device = LavaBackend()` and any KA-based GPU code runs on Vulkan.

2. **A graphics shader backend** where vertex, fragment, geometry, and tessellation shaders are written in Julia (not GLSL). Pass regular Julia functions to `Rasterizer(vertex=f, fragment=g)`.

3. **A hardware-accelerated ray tracing backend** with raygen, closest-hit, any-hit, miss, intersection shaders using `VK_KHR_ray_tracing_pipeline`. Julia functions compile to RT shader stages with real hardware BVH traversal.

**Unified buffer model**: A single `LavaArray{T,N}` is usable across all shader stages. Compute kernels fill geometry, vertex shaders render it, ray tracing shaders trace against it. No copies, no format conversion.

Cross-platform: NVIDIA, AMD, Intel, Apple (MoltenVK), software (lavapipe).

## API Stability

The **KernelAbstractions.jl / GPUArrays.jl interface** (`LavaBackend`, `LavaArray`, `@kernel`, broadcasting, reductions) is considered stable. This is the standard Julia GPU ecosystem API and behaves like any other backend.

The **graphics** and **ray tracing** APIs (`Rasterizer`, `GraphicsPipeline`, `RayTracingPipeline`, `HardwareAccel`, etc.) are functional but still evolving. Expect breaking changes as we iterate on the design.

## Quick Start

```julia
using Lava, KernelAbstractions

# Compute: drop-in replacement for CUDABackend() or ROCBackend()
backend = LavaBackend()

@kernel function vadd!(C, A, B)
    i = @index(Global)
    C[i] = A[i] + B[i]
end

a = LavaArray(rand(Float32, 1024))
b = LavaArray(rand(Float32, 1024))
c = LavaArray{Float32}(undef, 1024)

vadd!(backend)(c, a, b; ndrange=1024)
@assert Array(c) ≈ Array(a) .+ Array(b)
```

```julia
# Graphics: Julia functions as shaders, no GLSL
pipeline = Rasterizer(
    vertex   = (pos, mvp) -> set_position!(mvp * pos),
    fragment = (color,)   -> color,
)
draw!(pipeline, framebuffer; vertices=mesh, uniforms=params)
```

```julia
# Hardware Ray Tracing: Julia materials on RT hardware
hw = HardwareAccel(scene.accel)
trace_closest_hits!(hits, rays, hw)
```

## Architecture

Lava.jl uses a **custom LLVM IR to SPIR-V emitter written in Julia** (~6,000 lines). This is not a wrapper around `llc` or the SPIRV-LLVM-Translator. It reads the LLVM module via LLVM.jl and emits SPIR-V directly, giving full control over all shader stages and Vulkan extensions.

```
Julia function
  |  GPUCompiler.jl
LLVM IR (via LLVM.jl, LLVM 18)
  |  LLVM passes: GEP lifting, StructurizeCFG, Vulkan preparation
  |  Custom SPIR-V emitter (~40 LLVM opcodes to SPIR-V)
  |  spirv-val validation
VkShaderModule -> VkPipeline -> GPU dispatch
```

**Why not llc?** The LLVM SPIR-V backend crashes on geometry shaders, rejects ray tracing extensions, and doesn't support key Vulkan extensions (`SPV_KHR_physical_storage_buffer`, `SPV_KHR_variable_pointers`). These are missing C++ instruction selection patterns with no upstream fix planned. The custom emitter handles all shader stages from day one.

## Features

### Compute
- **GPUArrays.jl**: ~99.9% test suite pass (9,578 passed, 26 failed across 27 groups)
- **KernelAbstractions.jl**: Full `LavaBackend <: KA.GPU` with `@kernel`, `@index`, `@localmem`, `@synchronize`
- **Atomix.jl**: Int32/UInt32/Float32 atomics (add, sub, and, or, xor, xchg, min, max; Float32 add via CAS loop)
- **Complex structs**: Deeply nested types, NTuple fields, multi-field structs pass through BDA without corruption
- **Hikari raytracer**: Full wavefront volume path tracer (93 source files, 20 materials, 6+ kernel stages) renders correctly

### Graphics
- Vertex, fragment, geometry, tessellation shaders from plain Julia functions
- `Rasterizer`, `TrianglePipeline`, `LinePipeline` high-level APIs
- `RenderWindow` with GLFW + swapchain, `LavaFramebuffer` for offscreen
- `LavaTexture2D`, `LavaSampler` with descriptor set management
- Blend modes, culling, depth testing, topology selection

### Ray Tracing
- `VK_KHR_ray_tracing_pipeline` + `VK_KHR_acceleration_structure`
- BLAS/TLAS build from triangle meshes
- `RayTracingPipeline(raygen=f, closest_hit=g, miss=h)` with Julia functions as RT shaders
- `HardwareAccel` integration with Hikari/Raycore for drop-in HW RT
- Multi-field payloads, any-hit shaders for alpha transparency
- Shadow rays with multi-round host dispatch

### Performance
- **Dispatch overhead**: 1.8us per dispatch (vs AMDGPU 8.8us), 11x improvement via `@generated` arg packing, pipeline cache, ghost type optimization
- **Throughput**: 2x faster than AMDGPU on single-dispatch 10M-element kernels
- **Hikari 10spp 1200x900**: ~1.42s (vs AMDGPU ~1.69s, 16% faster)
- **Batched dispatch**: Multiple kernels recorded into a single command buffer, submitted together
- **Unified/BAR memory**: Auto-detected for small buffers, eliminates staging copies

### Kernel Arguments: Buffer Device Address (BDA)
All kernel arguments are packed into a device-memory buffer with a single `i64` push constant pointing to the BDA. No descriptor sets per buffer, no 256-byte push constant limit. One push constant, unlimited arguments.

## Source Structure

```
src/
├── Lava.jl                          # Module, exports, debugging API
├── compiler/
│   ├── compilation.jl               # Pipeline orchestration, LLVM pass coordination
│   ├── target.jl                    # GPUCompiler target (LavaCompilerTarget)
│   ├── entry_wrapper.jl             # BDA argument buffer wrapping
│   ├── passes/
│   │   ├── lift_geps.jl             # Byte-offset GEP to typed GEP (3 passes)
│   │   ├── structurize_cfg.jl       # CFG fixup + StructurizeCFGPass
│   │   ├── prepare_vulkan.jl        # 8 Vulkan IR transforms, shared mem flattening
│   │   └── lower_intrinsics.jl      # Math/barrier/thread-index lowering
│   └── spirv/
│       ├── module.jl                # SPIR-V module builder, ID allocation, serialization
│       ├── types.jl                 # LLVM to SPIR-V type mapping, opaque pointer recovery
│       ├── emit.jl                  # Core emitter: ~40 opcodes, control flow, decorations
│       ├── graphics.jl              # Graphics shader entry points
│       └── raytracing.jl            # RT shader stages, OpTraceRayKHR
├── device/
│   ├── math.jl                      # Float32 GLSL.std.450 + Float64 fallbacks
│   ├── quirks.jl                    # @device_override for GPU-safe replacements
│   ├── atomics.jl                   # Atomix.jl integration, CAS loops
│   ├── rt_intrinsics.jl             # RT builtins (trace_ray, payload I/O)
│   └── gfx_intrinsics.jl            # Graphics builtins (vertex_index, frag_coord, etc.)
├── runtime/
│   ├── device.jl                    # Vulkan device/queue creation, feature detection
│   ├── memory.jl                    # Buffer allocation, BDA, staging, mapped memory
│   ├── command.jl                   # Command buffer pool, batched submission
│   ├── launch.jl                    # lava_launch!, arg buffer pool, vk_flush!
│   ├── pipeline.jl                  # VkShaderModule + pipeline caching
│   ├── errors.jl                    # LavaError, validation layer capture
│   └── intrinsics.jl                # Runtime intrinsic helpers
├── array/
│   ├── lavaarray.jl                 # LavaArray{T,N} + LavaDeviceArray
│   ├── gpuarrays.jl                 # GPUArrays.jl interface, BroadcastStyle
│   ├── ka_backend.jl                # KA LavaBackend, @kernel/@index/@localmem
│   └── mapreduce.jl                 # GPU mapreducedim! with subgroup reductions
├── graphics/
│   ├── types.jl                     # BlendMode, CullFace, Topology, DepthMode, etc.
│   ├── api.jl                       # GraphicsPipeline, draw!, blit!, present_frame!
│   ├── pipeline.jl                  # CompiledGraphicsPipeline, draw recording
│   ├── framebuffer.jl               # LavaFramebuffer, render targets
│   ├── window.jl                    # RenderWindow, swapchain, GLFW
│   └── textures.jl                  # LavaTexture2D, LavaSampler
└── raytracing/
    ├── acceleration.jl              # BLAS/TLAS build, HardwareAccel
    ├── pipeline.jl                  # RT pipeline, SBT, vkCmdTraceRaysKHR
    ├── shaders.jl                   # RayTracingPipeline high-level API
    └── raycore_compat.jl            # trace_closest_hits!, Hikari integration
```

## Dependencies

| Package | Role |
|---------|------|
| GPUCompiler.jl | Julia to LLVM IR compilation |
| LLVM.jl (v9.4, LLVM 18) | LLVM IR manipulation and passes |
| Vulkan.jl (v0.6) | Vulkan API bindings |
| GPUArrays.jl | AbstractGPUArray interface |
| KernelAbstractions.jl | Backend-agnostic kernel API |
| Adapt.jl | Host/device type conversion |
| Atomix.jl | Atomic operations |
| SPIRV_Tools_jll | spirv-val, spirv-dis |
| GeometryBasics.jl | Vec/Mat/Point types for geometry math |
| GLFW | Window creation for graphics |

## Hardware

Tested on:
- AMD RX 7900 XTX (Vulkan 1.4, RADV)
- NVIDIA RTX 3070 Laptop (Vulkan 1.3)
- lavapipe (software, CI)

Requires Vulkan 1.2+ with `bufferDeviceAddress` and `variablePointers` features.

## Test Suite

```
Total: 9,578 passed, 26 failed, 0 errors (27 test groups, 4 skipped)
```

All groups 100% pass except:
- `linalg/norm`: 673/696 (0.5/1.5-norm precision edge cases)
- `statistics`: 75/78 (`mean(sin, Float16)` tree reduction order)

Run tests:
```julia
using Pkg; Pkg.test("Lava")
```

## Debugging

```julia
using Lava

# Memory usage
gpu_memory_usage()  # -> (live_bytes=..., live_buffers=..., ...)

# Full runtime state
dump_state()

# Dispatch logging
set_dispatch_logging!(true)
# ... run kernels ...
get_dispatch_log()

# Reset device after error
vk_reset_device!()
```

## Benchmarks

Render benchmarks on AMD RX 7900 XTX / Ryzen 9 7900X, using [Hikari](https://github.com/SimonDanisch/Hikari.jl) wavefront volume path tracer via [RayMakie](https://github.com/SimonDanisch/RayMakie.jl). Full benchmark suite in [RayDemo/benchmark/](https://github.com/SimonDanisch/RayDemo/tree/sd/benchmarks/benchmark).

### Render Time (seconds, lower is better)

| Scene | Lava HW RT | Lava SW | AMDGPU | pbrt-v4 CPU |
|-------|-----------|---------|--------|-------------|
| Crown (500x700, 16spp) | **0.39** | 0.75 | 1.08 | 12.5 |
| Bunny Cloud (960x540, 8spp) | **1.74** | 1.78 | 2.25 | 10.2 |
| Killeroo Gold (684x513, 32spp) | **0.33** | 0.37 | 0.76 | 7.5 |
| Materials (1200x900, 10spp) | **0.65** | 0.76 | 1.53 | 3.5 |
| Black Hole (800x450, 32spp) | 1.64 | **1.68** | 4.25 | — |

Lava SW is **1.4-2.1x faster than AMDGPU** across all scenes. Hardware RT adds another **1.1-2.3x** on geometry-heavy scenes (Crown, Killeroo Gold). Compared to pbrt-v4 on 24 CPU threads: **5-22x faster**.

### AcceleratedKernels (100M elements, ms, lower is better)

| Operation | Lava | AMDGPU |
|-----------|------|--------|
| sort/UInt32 | 138.0 | **106.7** |
| sort/Float32 | 148.2 | **120.9** |
| reduce/UInt32 | **0.92** | 1.22 |
| reduce/Float32 | **0.92** | 1.08 |
| accumulate/Float32 | 7.52 | **6.78** |
| map/Float32 (2x) | **1.36** | 1.61 |
| map/Float32 (sin) | **1.36** | 3.99 |
| mapreduce/Float32 (sin) | **1.06** | 1.53 |
| sortperm/UInt32 | 309.7 | **226.7** |

Lava wins on compute-bound operations (reduce, map, mapreduce — up to **2.9x** on map/sin). AMDGPU wins on memory-bound operations (sort, sortperm) where its native HIP driver has an edge.

## Status

Compute (GPUArrays + KernelAbstractions) works and is well tested. Graphics shaders and hardware ray tracing are functional. Multi-vendor CI, docs, and further performance work are ongoing.
