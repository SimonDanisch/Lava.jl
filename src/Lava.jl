module Lava

export LavaArray, LavaBackend, LavaBuffer, LavaDeviceArray, alloc_index_buffer
export BatchQueue
export CompilationResult, lava_compile, optimize_spirv
# Debugging & diagnostics
export vk_reset_device!, dump_state, gpu_memory_usage, allocate_batch_queue!
export set_dispatch_logging!, get_dispatch_log

# Graphics exports
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export RenderWindow, LavaFramebuffer, WindowTarget, OffscreenTarget
export CompiledGraphicsPipeline, LavaGfxShader
export draw!, blit!, present_frame!, acquire_next_image!, readback_framebuffer, readback_window
export vk_begin_pass!, vk_draw_in_pass!, vk_draw_indexed_in_pass!, vk_end_pass!
export pack_gfx_args, ensure_compiled!, transition_image!
export LavaTexture2D, LavaTexture1D, LavaSampler, SampledTexture, LavaTexture
export TextureBindings, bind_textures
# Graphics types
export ShaderStage, VertexStage, FragmentStage, GeometryStage, TessControlStage, TessEvalStage
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied
export CullFace, NoCull, CullBack, CullFront
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList, PatchList, LineListAdjacency
export DepthMode, DepthLess, DepthLessEq, DepthGreater, DepthAlways, DepthOff
export GeometryConfig, TessConfig
export TessSpacing, EqualSpacing, FractionalEvenSpacing, FractionalOddSpacing
export TessWinding, WindingCW, WindingCCW
export TessDomain, TessTriangles, TessQuads, TessIsolines
# Graphics device intrinsics
export vertex_index, instance_index
export frag_coord, frag_coord_x, frag_coord_y, frag_coord_xy
export dFdx, dFdy
export front_facing
export set_position!, set_point_size!
export gfx_output, gfx_input, gfx_output_flat, gfx_input_flat
export emit_vertex!, end_primitive!, invocation_id, primitive_id_in
export geom_input, geom_input_position
export tess_coord, tess_coord_uvw, set_tess_level_outer!, set_tess_level_inner!
export sample_texture_2d, GfxTexture2D

# Ray tracing exports
export HardwareAccel, RTRay, RTHitResult, ASBuildContext, as_build
export trace_closest_hits!, trace_closest_hits_indirect!, RayTracingPipeline, trace_rays!, trace_rays_indirect!
export set_anyhit_pipeline!, trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export lava_rt_ignore_intersection, lava_rt_terminate_ray

import Serialization
using Vulkan
using GPUCompiler
using LLVM
using LLVM: API
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix
using SPIRV_Tools_jll
using LinearAlgebra
using GeometryBasics
import GLFW

# ---- Graphics types (pure Julia, no Vulkan dependency) ----
include("graphics/types.jl")

# ---- Error infrastructure ----
include("runtime/errors.jl")

# ---- SPIR-V module builder ----
include("compiler/spirv/module.jl")

# ---- SPIR-V emitter stages ----
include("compiler/spirv/types.jl")
include("compiler/spirv/emit.jl")

# ---- LLVM passes ----
include("compiler/passes/lift_geps.jl")
include("compiler/passes/structurize_cfg.jl")
include("compiler/passes/prepare_vulkan.jl")
include("compiler/passes/lower_intrinsics.jl")
# shared_memory.jl — consolidated into prepare_vulkan.jl

# ---- Compiler (depends on emitter + passes) ----
include("compiler/target.jl")
include("compiler/entry_wrapper.jl")
include("compiler/compilation.jl")
# Planned files consolidated into emit.jl (4431 lines) and compilation.jl:
#   sourcemap.jl, intrinsics.jl, control_flow.jl, decorations.jl, compute.jl
include("compiler/spirv/raytracing.jl")    # RT stages, OpTraceRayKHR, payload handling
include("compiler/spirv/graphics.jl")     # Graphics stages, I/O variables, emit_vertex

# ---- Device functions ----
# device/array.jl — LavaDeviceArray defined in array/lavaarray.jl
include("device/math.jl")
include("device/quirks.jl")
include("device/rt_intrinsics.jl")  # RT builtins + trace_ray intrinsic
include("device/gfx_intrinsics.jl")  # Graphics builtins + shader I/O intrinsics
# atomics.jl is included after lavaarray.jl (needs LavaDeviceArray)

# ---- Vulkan runtime ----
include("runtime/device.jl")
include("runtime/memory.jl")
include("runtime/pipeline.jl")
include("runtime/command.jl")
include("runtime/intrinsics.jl")

# ---- Array interface (before launch.jl because it references LavaArray) ----
include("array/lavaarray.jl")
include("device/atomics.jl")  # needs LavaDeviceArray from lavaarray.jl

# ---- Launch API (depends on LavaArray and LavaBuffer) ----
include("runtime/launch.jl")
# runtime/sync.jl — sync handled via vk_flush!() in launch.jl

# ---- KernelAbstractions backend ----
include("array/ka_backend.jl")
include("array/gpuarrays.jl")
include("array/mapreduce.jl")

# ---- Phase 2: Graphics ----
include("graphics/window.jl")         # RenderWindow, swapchain, present
include("graphics/framebuffer.jl")   # LavaFramebuffer, RenderTarget, image memory helpers
include("graphics/pipeline.jl")      # CompiledGraphicsPipeline, draw recording
include("graphics/textures.jl")      # LavaTexture2D, LavaSampler, descriptor sets
include("graphics/api.jl")           # GraphicsPipeline, draw!, blit!, present_frame!

# ---- Phase 2: Ray Tracing ----
include("raytracing/acceleration.jl")  # BLAS/TLAS build
include("raytracing/pipeline.jl")      # RT pipeline + SBT + dispatch
include("raytracing/shaders.jl")       # High-level RayTracingPipeline API
include("raytracing/raycore_compat.jl") # HardwareAccel + trace_closest_hits!

# ---- Default settings ----
# Disable scalar indexing by default (GPU arrays should not be accessed element-by-element)
GPUArraysCore.allowscalar(false)

# ---- Debugging & Diagnostics API ----

"""
    gpu_memory_usage() -> NamedTuple

Return current GPU memory usage statistics.
"""
function gpu_memory_usage()
    (live_bytes = GPU_LIVE_BYTES[],
     LIVE_BUFFERS = length(LIVE_BUFFERS),
     deferred_frees = length(DEFERRED_FREES),
     ARG_SLABS = length(ARG_SLABS),
     pipelines_cached = length(PIPELINE_CACHE),
     kernels_cached = length(LINKED_KERNEL_CACHE))
end

"""
    dump_state(; io::IO=stdout)

Print a comprehensive summary of Lava.jl runtime state for debugging.
"""
function dump_state(; io::IO=stdout)
    ctx = VK_CONTEXT_REF[]
    println(io, "=== Lava.jl State ===")
    println(io, "Device: ", ctx === nothing ? "not initialized" : ctx.device_name)
    println(io, "Device lost: ", DEVICE_LOST[])
    mem = gpu_memory_usage()
    live_mb = mem.live_bytes ÷ (1024 * 1024)
    println(io, "GPU memory: $(live_mb) MiB in $(mem.LIVE_BUFFERS) buffers ($(mem.deferred_frees) deferred)")
    println(io, "Pipelines cached: $(mem.pipelines_cached) (max $(MAX_PIPELINE_CACHE_SIZE[]))")
    println(io, "Kernels cached: $(mem.kernels_cached) (max $(MAX_KERNEL_CACHE_SIZE[]))")
    println(io, "Arg slabs: $(mem.ARG_SLABS) (slab_idx=$(ARG_SLAB_IDX[]), offset=$(ARG_SLAB_OFFSET[]))")
    if ctx !== nothing
        println(io, "Free batches: $(length(ctx.free_batches))")
        println(io, "Free cmd bufs: $(length(ctx.free_cmd_bufs))")
        println(io, "CB split threshold: $(CB_SPLIT_THRESHOLD[])")
    end
    println(io, "Flushes: $(FLUSH_COUNTER[])")
    println(io, "Total dispatches: $(TOTAL_DISPATCH_COUNTER[])")
    println(io, "Dispatch logging: $(DISPATCH_LOGGING_ENABLED[])")
    if !isempty(DISPATCH_LOG)
        println(io, "Last dispatch: ", last(DISPATCH_LOG))
    end
    return nothing
end

function __init__()
    # Reset runtime state that should not survive precompilation.
    # Vulkan handles serialized into the pkgimage are invalid in a new process.
    VK_CONTEXT_REF[] = nothing
    DEVICE_LOST[] = false
    FLUSH_COUNTER[] = 0
    TOTAL_DISPATCH_COUNTER[] = 0
    LAST_DISPATCH_INFO[] = ""
    PREV_DISPATCH_INFO[] = ""
    empty!(DISPATCH_LOG)
    empty!(ENABLED_OPTIONAL_FEATURES)
    empty!(COMPILER_CONFIGS)
    empty!(PIPELINE_CACHE)
    empty!(LINKED_KERNEL_CACHE)
    empty!(LIVE_BUFFERS)
    empty!(DEFERRED_FREES)
    GPU_LIVE_BYTES[] = 0
    GPU_BYTES_SINCE_LAST_GC[] = 0
    empty!(POOL_BLOCKS)
    for fl in POOL_FREE_LISTS; empty!(fl); end
    empty!(ARG_SLABS)
    ARG_SLAB_IDX[] = 1
    ARG_SLAB_OFFSET[] = 0
    ARG_ALLOC_COUNT[] = 0
    empty!(ARG_BUFFERS)
    ARG_BUFFER_IDX[] = 0
    empty!(INDIRECT_SLABS)
    INDIRECT_SLAB_IDX[] = 1
    INDIRECT_SLAB_OFFSET[] = 0
    empty!(INDIRECT_BUFFERS)
    INDIRECT_BUFFER_IDX[] = 0
    CMD_PIPELINE_BARRIER_FPTR[] = C_NULL
    init_pipeline_thread!()

    # Mark device as lost during shutdown so GC finalizers don't call into
    # the Vulkan driver after it's been torn down. The atexit hook runs
    # before Julia's global finalizer sweep.
    atexit() do
        DEVICE_LOST[] = true
        VK_CONTEXT_REF[] = nothing
    end
end

# ── Precompilation hints ──
# Precompile hot Julia-side codepaths via explicit method hints.
# IMPORTANT: We must NOT initialize any Vulkan device or allocate GPU memory here.
# Stale Vulkan handles serialized into the pkgimage would segfault on load.
# Instead, use precompile() to force type inference on key methods.

using PrecompileTools

@setup_workload begin
    @compile_workload begin
        # Safe: compiler config creation (pure Julia, no Vulkan)
        lava_compiler_config(; workgroup_size=(64, 1, 1))
        lava_compiler_config(; workgroup_size=(256, 1, 1))

        # Safe: hash computation for cache keys (pure Julia)
        hash((typeof(identity), Tuple{Ptr{Float32}, Ptr{Float32}}, (64, 1, 1)))

        # Safe: LavaAdaptor and adapt paths (pure Julia type dispatch)
        Adapt.adapt(LavaAdaptor(), Float32(1.0))
        Adapt.adapt(LavaAdaptor(), Int32(1))

        # Safe: GPUCompiler target configuration (pure Julia)
        GPUCompiler.SPIRVCompilerTarget(; backend=:llvm, validate=false, supports_fp64=true)
    end
end

# Explicit precompile hints for hot methods that involve complex type parameters.
# These force Julia to cache the type inference results without executing anything.
let
    # Arg packing (@generated functions) - common kernel argument types
    for T in (Float32, Int32, UInt32)
        precompile(Tuple{typeof(hash), Tuple{Type, Type, NTuple{3,Int}}})
    end

    # LavaArray operations
    precompile(Tuple{typeof(Adapt.adapt), LavaAdaptor, Vector{Float32}})

    # Disk cache paths
    precompile(Tuple{typeof(lava_disk_cache_dir)})
    precompile(Tuple{typeof(lava_disk_cache_key), Core.MethodInstance, NTuple{3,Int}})
end

end # module Lava
