module Lava

export LavaArray, LavaBackend, LavaDeviceArray, alloc_index_buffer
export AcceleratedMatrix, MatrixA, MatrixB, Accumulator, CoopMat, Scalar
export matriximpl
export BatchQueue
# Freezing kernels: `@setup_workload` is PrecompileTools', re-exported so a
# package needs one import to get both halves of the workload.
export @setup_workload, @compile_workload
export CompilationResult, lava_compile, optimize_spirv
# Debugging & diagnostics
export vk_reset_device!, dump_state, gpu_memory_usage, trim_gpu_pool!,
       allocate_batch_queue!
export ExternalImage, memoryfd
# Device capability queries: API, but not exported — a kernel library reaches
# them as `Lava.shader_core_count()`, alongside `Lava.device_subgroup_size()`.
public shader_core_count, shader_warps_per_sm, max_shared_memory, DeviceCompute
export set_dispatch_logging!, get_dispatch_log
export concurrent_dispatch_group, concurrent_indirect_group
export enable_gpu_av, disable_gpu_av, verify_gpu_av, activate_all_debugging

# Graphics exports
export GraphicsPipeline, Rasterizer, TrianglePipeline, LinePipeline
export RenderWindow, LavaFramebuffer, WindowTarget, OffscreenTarget
export CompiledGraphicsPipeline, LavaGfxShader
export draw!, blit!, present_frame!, acquire_next_image!, sync_swapchain!,
       readback_framebuffer, readback_window
export copy_framebuffer!
export vk_begin_pass!, vk_draw_in_pass!, vk_draw_indexed_in_pass!, vk_set_viewport!, vk_end_pass!
export vk_draw_indirect_in_pass!, DrawIndirectCommand, indirect_buffer
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

# Compute kernels
export write_grain_instances_kernel,
       quat_to_rot3x3, build_4x3, build_4x3_pervec

# Ray tracing exports
export LavaInstanceRecord, identity_transform, Mat3x4f
export GeometryType, TrianglesGeometry, AABBsGeometry, AABB
# Re-export Raycore.Ray so `using Lava` users get Ray without ambiguity
export Ray
export HardwareAccel, RTRay, RTHitResult, ASBuildContext, as_build
export refit_tlas!
export trace_closest_hits!, trace_closest_hits_indirect!, RayTracingPipeline, trace_rays!, trace_rays_indirect!
export set_anyhit_pipeline!, trace_closest_hits_anyhit!, trace_closest_hits_anyhit_indirect!
export lava_rt_ignore_intersection, lava_rt_terminate_ray
# SER (SPV_NV_shader_invocation_reorder)
export lava_rt_hit_object_trace_ray, lava_rt_reorder_thread, lava_rt_hit_object_execute_shader
export HWTLAS, HWAdaptedAccel
# P4 narrow-phase convex shapes
export ConvexShape, UnitCube, support
# P4.2 GJK overlap test
export gjk, GJKResult
# P4.3 EPA penetration recovery
export epa, EPAResult
# P4.4 narrow-phase kernel composing gjk + epa
export narrow_phase_kernel, NO_CONTACT
# P4.5 contact record + per-grain contact buffer + fused narrow-phase
export ContactRecord, narrow_phase_contacts_kernel

import Serialization
import PrecompileTools
using PrecompileTools: @setup_workload
using Vulkan
using GPUCompiler
using Raycore: Ray
using LLVM
using LLVM: API
using GPUArrays
using GPUArraysCore
using KernelAbstractions
using Adapt
using Atomix
using SPIRV_Tools_jll
using LinearAlgebra
using StaticArrays
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

# ---- Compiler phase timing ----
include("compiler/phase_timer.jl")

# ---- LLVM passes ----
include("compiler/passes/lift_geps.jl")
include("compiler/passes/retype_allocas.jl")
include("compiler/passes/cfg_utils.jl")
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
include("compiler/spirv/rayquery.jl")     # VK_KHR_ray_query capabilities + emission
include("compiler/spirv/coopmat.jl")      # SPV_KHR_cooperative_matrix emission
include("compiler/spirv/graphics.jl")     # Graphics stages, I/O variables, emit_vertex

# ---- Device functions ----
# device/array.jl — LavaDeviceArray defined in array/lavaarray.jl
include("device/math.jl")
include("device/quirks.jl")
include("device/rt_intrinsics.jl")  # RT builtins + trace_ray intrinsic
include("device/ray_query_intrinsics.jl")  # lava_ray_query_init
include("device/gfx_intrinsics.jl")  # Graphics builtins + shader I/O intrinsics
include("device/printf.jl")          # @lava_printf → NonSemantic.DebugPrintf
# atomics.jl is included after lavaarray.jl (needs LavaDeviceArray)

# ---- Vulkan runtime ----
# Types before the context that names them. `VkContext` holds its per-device
# state in concrete fields, so every struct in those fields has to exist first —
# and only the `struct` blocks moved, because include order constrains types and
# not methods.
include("runtime/coretypes.jl")
include("runtime/device.jl")
include("runtime/memory.jl")
include("runtime/pipeline.jl")
include("runtime/command.jl")
include("runtime/intrinsics.jl")

# ---- Array interface (before launch.jl because it references LavaArray) ----
include("array/lavaarray.jl")
include("device/atomics.jl")  # needs LavaDeviceArray from lavaarray.jl
include("device/subgroup.jl") # subgroup / group-non-uniform intrinsics
include("device/acceleratedmatrix.jl")     # AcceleratedMatrix: the user-facing type
include("device/coopmat_intrinsics.jl")    # its llvmcall stubs
include("array/pin_leaves.jl") # @generated walker that pins LavaArray leaves per batch

# ---- Launch API (depends on LavaArray / LavaDeviceArray) ----
include("runtime/launch.jl")
# Frozen kernel cache — `launch.jl` calls into it, and it calls `link_kernel`
# back; both are resolved at call time, so the include order is free.
include("runtime/frozen_cache.jl")
# The workload macros sit on top of the frozen cache and PrecompileTools.
include("runtime/workload.jl")
# Pipeline cache persistence — depends on lava_disk_cache_dir from launch.jl;
# referenced by VkContext constructor (forward at include time, resolved at call time)
include("runtime/pipeline_cache.jl")
# runtime/sync.jl — sync handled via vk_flush!() in launch.jl

# ---- KernelAbstractions backend ----
include("array/ka_backend.jl")
include("array/gpuarrays.jl")
include("array/gemm.jl")       # mul! for LavaArray — needs AnyLavaArray from gpuarrays.jl
include("array/mapreduce.jl")

# ---- Debug / validation API (needs LavaArray + KA backend) ----
include("runtime/debug.jl")
include("runtime/external.jl")

# ---- Hardware video decode (VK_KHR_video_decode_queue) ----
include("runtime/video.jl")
"""
    decode_h264_gpu(annexb::Vector{UInt8}; maxframes) -> (width, height, Vector{LavaArray{UInt8,2}})

Hardware-decode an H.264 Annex-B elementary stream on the GPU (VK_KHR_video_decode),
keeping every decoded luma (Y) plane GPU-resident: each frame is returned as a
device-local `LavaArray{UInt8,2}` (cropped to the display size), never touching host
memory. Feed these straight into Lava kernels / the motion tracker. Requires a device
created with video decode support (`vk_context().video_decode_available`).
"""
function decode_h264_gpu(annexb::Vector{UInt8}; kw...)
    w, h, ys, _ = VideoDecode.decode_h264(vk_context(), annexb; kw...)
    return (w, h, ys)
end

"""
    decode_h264_nv12(annexb; maxframes) -> (width, height, ys, uvs)

Like [`decode_h264_gpu`] but also returns the chroma plane: `ys` are the luma
`LavaArray{UInt8,2}` (width×height) and `uvs` the interleaved-U/V chroma
`LavaArray{UInt8,2}` (width×(height÷2)) for each frame — the two planes of NV12,
GPU-resident. Convert to RGB on-GPU with GPUFiltering's `nv12torgb!`.
"""
function decode_h264_nv12(annexb::Vector{UInt8}; kw...)
    w, h, ys, uvs = VideoDecode.decode_h264(vk_context(), annexb; chroma=true, kw...)
    return (w, h, ys, uvs)
end

"""
    decode_h264_luma(annexb::Vector{UInt8}; maxframes) -> (width, height, Vector{Matrix{UInt8}})

Like [`decode_h264_gpu`] but downloads each decoded luma plane to a host
`Matrix{UInt8}` (grayscale = the NV12 Y plane). Used to validate the GPU decoder
against a reference (e.g. ffmpeg).
"""
function decode_h264_luma(annexb::Vector{UInt8}; kw...)
    w, h, frames = decode_h264_gpu(annexb; kw...)
    return (w, h, Matrix{UInt8}[Array(f) for f in frames])
end

# ---- Profiling (kernel SPIR-V stats + per-dispatch GPU timing) ----
include("runtime/profiling.jl")

# ---- Phase 2: Graphics ----
include("graphics/window.jl")         # RenderWindow, swapchain, present
include("graphics/framebuffer.jl")   # LavaFramebuffer, RenderTarget, image memory helpers
include("graphics/pipeline.jl")      # CompiledGraphicsPipeline, draw recording
include("graphics/textures.jl")      # LavaTexture2D, LavaSampler, descriptor sets
include("graphics/api.jl")           # GraphicsPipeline, draw!, blit!, present_frame!

# ---- Phase 2: Ray Tracing ----
include("raytracing/geometry_types.jl")  # GeometryType hierarchy + AABB struct
include("raytracing/instance_record.jl") # LavaInstanceRecord — 64-byte TLAS instance struct
include("raytracing/acceleration.jl")  # BLAS/TLAS build
include("raytracing/pipeline.jl")      # RT pipeline + SBT + dispatch
include("raytracing/shaders.jl")       # High-level RayTracingPipeline API
include("raytracing/raycore_compat.jl") # HardwareAccel + trace_closest_hits!
include("raytracing/hwtlas.jl")         # Lava.HWTLAS — concrete AbstractAccel
include("raytracing/convex_shape.jl")   # ConvexShape abstract + UnitCube + support (P4)
include("raytracing/gjk.jl")            # GJK overlap test + GJKResult (P4.2)
include("raytracing/epa.jl")            # EPA penetration recovery + EPAResult (P4.3)

# ---- Compute kernels ----
include("kernels/instance_writer.jl")   # write_grain_instances_kernel + helpers
include("kernels/narrow_phase.jl")      # narrow_phase_kernel composing gjk + epa (P4.4)

# ---- Default settings ----
# Disable scalar indexing by default (GPU arrays should not be accessed element-by-element)
GPUArraysCore.allowscalar(false)

# ---- Debugging & Diagnostics API ----

"""
    gpu_memory_usage() -> NamedTuple

Return current GPU memory usage statistics.
"""
function gpu_memory_usage()
    ctx = VK_CONTEXT_REF[]
    bq_deferred = 0
    n_arg_slabs = 0
    if ctx !== nothing
        bq = ctx.default_bq
        bq_deferred = length(bq.deferred_frees) + length(bq.deferred_as_frees)
        n_arg_slabs = length(bq.arg_slabs)
    end
    # Both counts come off the context now, so both are guarded by the same
    # `ctx !== nothing` as the queue fields above — a caller with no device gets
    # zeros rather than an error.
    n_pipelines = ctx === nothing ? 0 : length(ctx.caches.pipelines)
    n_kernels   = ctx === nothing ? 0 : length(ctx.caches.linked)
    (live_bytes = ctx === nothing ? 0 : gpu_live_bytes(ctx),
     live_buffers = ctx === nothing ? 0 : live_buffer_count(ctx),
     deferred_frees = bq_deferred,
     ARG_SLABS = n_arg_slabs,
     pipelines_cached = n_pipelines,
     kernels_cached = n_kernels)
end

"""
    dump_state(; io::IO=stdout)

Print a comprehensive summary of Lava.jl runtime state for debugging.
"""
function dump_state(; io::IO=stdout)
    ctx = VK_CONTEXT_REF[]
    println(io, "=== Lava.jl State ===")
    println(io, "Device: ", ctx === nothing ? "not initialized" : ctx.device_name)
    println(io, "Device lost: ", device_lost())
    mem = gpu_memory_usage()
    live_mb = mem.live_bytes ÷ (1024 * 1024)
    println(io, "GPU memory: $(live_mb) MiB in $(mem.live_buffers) buffers ($(mem.deferred_frees) deferred)")
    println(io, "Pipelines cached: $(mem.pipelines_cached) (max $(MAX_PIPELINE_CACHE_SIZE[]))")
    println(io, "Kernels cached: $(mem.kernels_cached)")
    if ctx !== nothing
        bq = ctx.default_bq
        println(io, "Arg slabs: $(length(bq.arg_slabs)) (slab_idx=$(bq.arg_slab_idx), offset=$(bq.arg_slab_offset))")
    end
    if ctx !== nothing
        bq = ctx.default_bq
        println(io, "Free batches: $(length(bq.free_batches))")
        println(io, "Free cmd bufs: $(length(bq.free_cmd_bufs))")
        ctx === nothing || println(io, "CB split threshold: $(ctx.default_bq.cb_split_threshold)")
    end
    println(io, "Flushes: $(ctx === nothing ? 0 : ctx.diag.flush_counter[])")
    println(io, "Total dispatches: $(ctx === nothing ? 0 : ctx.diag.total_dispatches[])")
    ctx === nothing || println(io, "Dispatch logging: $(ctx.diag.dispatch_logging)")
    dlog = ctx === nothing ? String[] : ctx.diag.dispatch_log
    if !isempty(dlog)
        println(io, "Last dispatch: ", last(dlog))
    end
    return nothing
end

function __init__()
    # Nothing to reset here any more. The counters and logs this used to zero
    # were module-level `Ref`s and `Vector`s, which meant a device crash during
    # precompilation serialised its wreckage into the pkgimage and poisoned every
    # later session. They are `ctx.diag` fields now, built fresh with the context,
    # so there is nothing that can survive into the image to clear.
    init_pipeline_thread!()

    # Mark device as lost during shutdown so GC finalizers don't call into
    # the Vulkan driver after it's been torn down. The atexit hook runs
    # before Julia's global finalizer sweep.
    atexit() do
        ctx = VK_CONTEXT_REF[]
        ctx === nothing || mark_device_lost!(ctx)
        VK_CONTEXT_REF[] = nothing
    end
end

end # module Lava
