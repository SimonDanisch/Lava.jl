module Lava

export LavaArray, LavaBackend, LavaBuffer, LavaDeviceArray
export CompilationResult, lava_compile, optimize_spirv
# Debugging & diagnostics
export vk_reset_device!, dump_state, gpu_memory_usage
export set_dispatch_logging!, get_dispatch_log

# Graphics exports
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export RenderWindow, LavaFramebuffer, WindowTarget, OffscreenTarget
export CompiledGraphicsPipeline, LavaGfxShader
export draw!, blit!, present_frame!, acquire_next_image!, readback_framebuffer
export vk_begin_pass!, vk_draw_in_pass!, vk_end_pass!
export pack_gfx_args, ensure_compiled!, transition_image!
export LavaTexture2D, LavaTexture1D, LavaSampler, SampledTexture, LavaTexture
export TextureBindings, bind_textures
# Graphics types
export ShaderStage, VertexStage, FragmentStage, GeometryStage, TessControlStage, TessEvalStage
export BlendMode, Opaque, AlphaBlend, Additive, Premultiplied
export CullFace, NoCull, CullBack, CullFront
export Topology, TriangleList, TriangleStrip, LineList, LineStrip, PointList, PatchList
export DepthMode, DepthLess, DepthLessEq, DepthGreater, DepthAlways, DepthOff
export GeometryConfig, TessConfig
export TessSpacing, EqualSpacing, FractionalEvenSpacing, FractionalOddSpacing
export TessWinding, WindingCW, WindingCCW
export TessDomain, TessTriangles, TessQuads, TessIsolines
# Graphics device intrinsics
export vertex_index, instance_index
export frag_coord, frag_coord_x, frag_coord_y, frag_coord_xy
export front_facing
export set_position!, set_point_size!
export gfx_output, gfx_input
export emit_vertex!, end_primitive!, invocation_id, primitive_id_in
export tess_coord, tess_coord_uvw, set_tess_level_outer!, set_tess_level_inner!
export sample_texture_2d

# Ray tracing exports
export HardwareAccel, RTRay, RTHitResult
export trace_closest_hits!, trace_closest_hits_indirect!, RayTracingPipeline, trace_rays!, trace_rays_indirect!
export set_anyhit_pipeline!, trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export lava_rt_ignore_intersection, lava_rt_terminate_ray

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
     live_buffers = length(_live_buffers),
     deferred_frees = length(DEFERRED_FREES),
     arg_slabs = length(_arg_slabs),
     pipelines_cached = length(_pipeline_cache),
     kernels_cached = length(_kernel_cache))
end

"""
    dump_state(; io::IO=stdout)

Print a comprehensive summary of Lava.jl runtime state for debugging.
"""
function dump_state(; io::IO=stdout)
    ctx = _vk_context[]
    println(io, "=== Lava.jl State ===")
    println(io, "Device: ", ctx === nothing ? "not initialized" : ctx.device_name)
    println(io, "Device lost: ", _device_lost[])
    mem = gpu_memory_usage()
    live_mb = mem.live_bytes ÷ (1024 * 1024)
    println(io, "GPU memory: $(live_mb) MiB in $(mem.live_buffers) buffers ($(mem.deferred_frees) deferred)")
    println(io, "Pipelines cached: $(mem.pipelines_cached) (max $(_max_pipeline_cache_size[]))")
    println(io, "Kernels cached: $(mem.kernels_cached) (max $(_max_kernel_cache_size[]))")
    println(io, "Arg slabs: $(mem.arg_slabs) (slab_idx=$(_arg_slab_idx[]), offset=$(_arg_slab_offset[]))")
    if ctx !== nothing
        println(io, "Free batches: $(length(ctx.free_batches))")
        println(io, "Free cmd bufs: $(length(ctx.free_cmd_bufs))")
        println(io, "CB split threshold: $(cb_split_threshold[])")
    end
    println(io, "Flushes: $(FLUSH_COUNTER[])")
    println(io, "Total dispatches: $(TOTAL_DISPATCH_COUNTER[])")
    println(io, "Dispatch logging: $(dispatch_logging_enabled[])")
    if !isempty(dispatch_log)
        println(io, "Last dispatch: ", last(dispatch_log))
    end
    return nothing
end

function __init__()
    # Reset runtime counters that should not survive precompilation.
    # These Ref values get serialized into the pkgimage — a device crash
    # during precompilation would permanently poison all future sessions.
    _device_lost[] = false
    FLUSH_COUNTER[] = 0
    TOTAL_DISPATCH_COUNTER[] = 0
    last_dispatch_info[] = ""
    prev_dispatch_info[] = ""
    empty!(dispatch_log)
    _init_pipeline_thread!()
end

end # module Lava
