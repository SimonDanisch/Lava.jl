# Architecture

## The pipeline

```
Julia function
  │  GPUCompiler.jl
LLVM IR (via LLVM.jl, LLVM 18)
  │  Custom LLVM passes:
  │   ├─ lift_geps        : byte-offset GEPs → typed GEPs (3 passes)
  │   ├─ structurize_cfg  : CFG fix-up + LLVM StructurizeCFGPass
  │   ├─ prepare_vulkan   : 8 Vulkan-specific transforms, shared-mem flattening
  │   └─ lower_intrinsics : math / barrier / thread-index lowering
  │  Custom SPIR-V emitter (~40 LLVM opcodes → SPIR-V)
  │  spirv-val validation
VkShaderModule → VkPipeline → GPU dispatch
```

The custom emitter is the architectural choice that makes the rest possible. Roughly 6,000 lines of Julia translate the post-pass LLVM module into SPIR-V directly via the LLVM.jl reflection API.

## Why not `llc`?

The LLVM project ships an SPIR-V backend (`llc -mtriple=spirv...`). Lava does not use it. Empirically (as of LLVM 21):

* **Geometry shaders crash `llc`** during instruction selection. No upstream fix is planned.
* **Ray tracing extensions are rejected** — `OpTraceRayKHR` and friends never made it past the prototype stage.
* **Key Vulkan extensions are missing** patterns: `SPV_KHR_physical_storage_buffer` (BDA) and `SPV_KHR_variable_pointers` are both incomplete. These are the features that make Lava's argument-passing model work.

The custom emitter handles all five graphics stages plus all RT stages from day one. `llc` output is still used as an oracle for cross-checking compute and vertex/fragment validation in the test suite.

## Source layout

```
src/
├── Lava.jl                          # Module entrypoint, exports, debugging API
├── compiler/
│   ├── compilation.jl               # Pipeline orchestration
│   ├── target.jl                    # GPUCompiler target definition
│   ├── entry_wrapper.jl             # BDA argument buffer wrapping
│   ├── passes/
│   │   ├── lift_geps.jl             # Byte-offset GEP to typed GEP (3 passes)
│   │   ├── structurize_cfg.jl       # CFG fixup + StructurizeCFGPass
│   │   ├── prepare_vulkan.jl        # 8 Vulkan IR transforms
│   │   └── lower_intrinsics.jl      # Math / barrier / thread index
│   └── spirv/
│       ├── module.jl                # SPIR-V module builder, ID allocation
│       ├── types.jl                 # LLVM ↔ SPIR-V type mapping
│       ├── emit.jl                  # Core emitter (~40 opcodes)
│       ├── graphics.jl              # Graphics shader entry points
│       └── raytracing.jl            # RT stages, OpTraceRayKHR
├── device/
│   ├── math.jl                      # Float32 GLSL.std.450 + Float64 fallbacks
│   ├── quirks.jl                    # @device_override for GPU-safe replacements
│   ├── atomics.jl                   # Atomix.jl integration, CAS loops
│   ├── rt_intrinsics.jl             # RT builtins
│   └── gfx_intrinsics.jl            # Graphics builtins
├── runtime/
│   ├── device.jl                    # Vulkan device/queue creation, feature detection
│   ├── memory.jl                    # Buffer allocation, BDA, staging, mapped memory
│   ├── command.jl                   # Command buffer pool, batched submission
│   ├── launch.jl                    # lava_launch!, arg buffer pool, vk_flush!
│   ├── pipeline.jl                  # VkShaderModule + pipeline caching
│   └── errors.jl                    # LavaError, validation layer capture
├── array/
│   ├── lavaarray.jl                 # LavaArray{T,N} + LavaDeviceArray
│   ├── gpuarrays.jl                 # GPUArrays.jl interface, BroadcastStyle
│   ├── ka_backend.jl                # LavaBackend, @kernel / @index / @localmem
│   └── mapreduce.jl                 # GPU mapreducedim! with subgroup reductions
├── graphics/
│   ├── api.jl                       # GraphicsPipeline, draw!, blit!, present_frame!
│   ├── pipeline.jl                  # CompiledGraphicsPipeline
│   ├── framebuffer.jl               # LavaFramebuffer
│   ├── window.jl                    # RenderWindow, swapchain, GLFW
│   └── textures.jl                  # LavaTexture2D, LavaSampler
└── raytracing/
    ├── acceleration.jl              # BLAS/TLAS build, HardwareAccel, HWTLAS
    ├── pipeline.jl                  # RT pipeline, SBT, vkCmdTraceRaysKHR
    ├── shaders.jl                   # RayTracingPipeline high-level API
    └── raycore_compat.jl            # trace_closest_hits!, Hikari integration
```

## Buffer Device Address (BDA)

The argument-passing model that lets Lava avoid the 256-byte push-constant limit and per-buffer descriptor sets. Every kernel entry is wrapped at compile time to expect one `i64` push constant — the BDA of an arg buffer. The wrapper unpacks the original arguments before calling the user function.

The arg buffer is allocated from a pool of mapped device-memory slabs, reused across dispatches, and replenished as needed. `vk_flush!` waits for outstanding command buffers and returns slabs to the pool.

## Memory and command-buffer lifecycle

* `LavaArray` is a thin wrapper around a `GPUArrays.DataRef{VkManagedBuffer}` plus an offset. The data ref handles reference counting; the underlying `VkBuffer` is freed when the last view drops *and* the last command buffer using it has retired.
* `CommandBatch` pins references to all GPU objects used during recording. Submission moves the batch into `bq.in_flight`. When the timeline semaphore signals completion, the batch is retired and its refs are released.
* Cross-queue synchronisation goes through timeline semaphores — no fence chains, no per-resource owner tracking.

## Compilation cache

Two layers, both transparent to the user:

* **In-memory by `MethodInstance`** — zero-cost after the first compile.
* **On-disk under `~/.julia/compiled/Lava/spirv/...`** — keyed by `IR hash × Vulkan device fingerprint`. Survives Julia restarts.

`Lava.clear_kernel_cache!()` evicts the in-memory layer; `Lava.clear_spirv_disk_cache!()` evicts the disk layer.

## Cross-platform notes

* **NVIDIA (RTX 30xx, 40xx)** — fully supported, including hardware RT.
* **AMD RADV (RX 7900, Radeon 8060S)** — fully supported. A known driver bug around long compute → RT mode switches is tracked in [Known Issues](known_issues.md).
* **Intel** — compute and graphics work; RT is driver-dependent.
* **lavapipe** — used for CI. Compute and graphics work; RT is software-emulated, slow but functional.
* **MoltenVK (macOS)** — compute works; graphics and RT depend on the MoltenVK feature level.
